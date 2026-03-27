"""
To be able to use instruct parameter you need faster-qwen3-tts version 0.2.5.

This version was not available on PyPI at the time of release, so you may need to install via git to use instruct parameter.
"""

import os
import time
import traceback
import inspect
import pyaudio
import numpy as np
from queue import Queue
from typing import Optional, Union, List, Tuple

try:
    import torch
    from faster_qwen3_tts import FasterQwen3TTS as _FasterQwen3TTS
except ImportError:
    torch = None
    _FasterQwen3TTS = None

from .base_engine import BaseEngine

# --- ANSI escape codes for debug styling ---
COLOR_BLUE   = "\033[94m"
COLOR_GREEN  = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN   = "\033[96m"
COLOR_RED    = "\033[91m"
COLOR_BOLD   = "\033[1m"
COLOR_RESET  = "\033[0m"

# Sentinel used to bypass audio file loading and trigger the pre‑primed cache.
# This string must never match a real file path on the target OS.
_CACHE_SENTINEL = "__preloaded_spk_emb__"


class FasterQwenVoice:
    """
    Represents a voice for the FasterQwenEngine.

    You must provide either a pre‑extracted speaker embedding (.pt file) or
    the reference audio and its transcript. The latter will trigger a one‑time
    extraction (saved to `speaker_pt` if provided) and then be cached.

    Args:
        name (str): Identifier for this voice.
        ref_audio (Optional[str]): Path to the reference audio file (.wav).
        ref_text (Optional[str]): Exact transcription of the reference audio.
        language (str): Target language for synthesis (e.g., "English", "German").
        instruct (Optional[str]): Instruction or emotion prompt.
        speaker_pt (Optional[str]): Path to a pre‑extracted speaker embedding (.pt).
    """
    def __init__(
            self,
            name: str,
            ref_audio: Optional[str] = None,
            ref_text: Optional[str] = None,
            language: str = "English",
            instruct: Optional[str] = None,
            speaker_pt: Optional[str] = None
    ):
        self.name = name
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.language = language
        self.instruct = instruct
        self.speaker_pt = speaker_pt

        if speaker_pt is None and (ref_audio is None or ref_text is None):
            raise ValueError(
                "FasterQwenVoice requires either speaker_pt or both ref_audio and ref_text."
            )

        # If only ref_audio/ref_text are given, we will extract the embedding on demand
        # and optionally save it if a path is supplied later via `extract_and_cache`.

    def __repr__(self):
        return (f"FasterQwenVoice(name={self.name}, lang={self.language}, "
                f"instruct={self.instruct}, speaker_pt={self.speaker_pt})")


class FasterQwenEngine(BaseEngine):
    """
    Real‑time TTS engine using faster‑qwen3‑tts with pre‑cached speaker embeddings.

    The speaker encoder is run exactly once per voice (at engine startup or when
    a new voice is set). The resulting embedding is stored in the model's internal
    cache, eliminating drift caused by repeated encoder calls.
    """

    def __init__(
            self,
            model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device: str = "cuda",
            voice: Optional[Union[str, FasterQwenVoice]] = None,
            chunk_size: int = 2,          # 1 = ~80ms TTFA, higher = less overhead
            xvec_only: bool = True,
            debug: bool = False,
            attn_implementation: str = "sdpa"   # "sdpa" works without compiling flash-attn
    ):
        """
        Initializes the FasterQwenTTS engine.

        Args:
            model_name (str): HF repo ID for the model.
            device (str): Device to run the model on (must be CUDA for CUDA graphs).
            voice (Optional[FasterQwenVoice]): The voice configuration.
            chunk_size (int): Number of codec steps per yielded chunk.
            xvec_only (bool): If True, uses only speaker embedding (cleaner language switching).
            debug (bool): If True, prints verbose streaming and latency metrics.
            attn_implementation (str): Attention implementation to use.
        """
        super().__init__()

        if _FasterQwen3TTS is None or torch is None:
            raise ImportError(
                "faster-qwen3-tts or torch is not installed. "
                "Please install them via: pip install faster-qwen3-tts torch"
            )

        self.model_name = model_name
        self.device = device
        self.chunk_size = chunk_size
        self.xvec_only = xvec_only
        self.debug = debug
        self.attn_implementation = attn_implementation

        self.queue = Queue()
        self.current_voice = None
        self._model = None
        self._is_warmed_up = False

        # Load the underlying model once
        if self.debug:
            print(f"\n{COLOR_CYAN}🧠 [FasterQwenEngine] Loading Model ({self.model_name}) on {self.device}...{COLOR_RESET}")

        load_start = time.time()
        self._model = _FasterQwen3TTS.from_pretrained(
            model_name,
            device=device,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation
        )
        if self.debug:
            print(f"{COLOR_GREEN}✓ Model loaded in {time.time() - load_start:.2f}s{COLOR_RESET}")

        # Prime the cache if a voice is provided
        if voice is not None:
            self.set_voice(voice)

        self.post_init()

    def post_init(self):
        self.engine_name = "faster_qwen3"

    def _extract_speaker_embedding(self, voice: FasterQwenVoice) -> "torch.Tensor":
        """
        Extract the speaker embedding from reference audio/text using the loaded model.
        This is a one‑time operation per voice; the result is cached in memory and
        optionally saved to voice.speaker_pt if that path is provided.

        Returns:
            torch.Tensor: The extracted embedding (on CPU).
        """
        if self.debug:
            print(f"{COLOR_YELLOW}🔄 [FasterQwenEngine] Extracting speaker embedding for voice '{voice.name}'...{COLOR_RESET}")

        # Create the prompt using the underlying qwen-tts model
        prompt_items = self._model.model.create_voice_clone_prompt(
            ref_audio=voice.ref_audio,
            ref_text=voice.ref_text,
            x_vector_only_mode=True,
        )
        spk_emb = prompt_items[0].ref_spk_embedding.cpu()

        # Save to file if a path is provided (for future reuse)
        if voice.speaker_pt:
            torch.save(spk_emb, voice.speaker_pt)
            if self.debug:
                print(f"{COLOR_GREEN}✓ Speaker embedding saved to {voice.speaker_pt}{COLOR_RESET}")

        return spk_emb

    def _prime_cache(self, voice: FasterQwenVoice):
        """
        Insert the speaker embedding into the model's internal cache under the sentinel key.
        If a speaker_pt file exists, load it; otherwise extract the embedding and
        save it to speaker_pt if that path is provided.
        """
        # Determine whether we need to load or extract
        if voice.speaker_pt and os.path.exists(voice.speaker_pt):
            if self.debug:
                print(f"{COLOR_BLUE}📦 [FasterQwenEngine] Loading speaker embedding from {voice.speaker_pt}...{COLOR_RESET}")
            spk_emb = torch.load(voice.speaker_pt, weights_only=True).to(self.device)
        else:
            # No existing file: extract the embedding
            # _extract_speaker_embedding will automatically save to voice.speaker_pt if it is provided
            if self.debug:
                if voice.speaker_pt:
                    print(f"{COLOR_YELLOW}🔄 [FasterQwenEngine] Extracting speaker embedding for voice '{voice.name}' (will save to {voice.speaker_pt})...{COLOR_RESET}")
                else:
                    print(f"{COLOR_YELLOW}🔄 [FasterQwenEngine] Extracting speaker embedding for voice '{voice.name}' (no save path)...{COLOR_RESET}")
            spk_emb = self._extract_speaker_embedding(voice).to(self.device)

        # Build the voice clone prompt dictionary used by the model
        vcp = dict(
            ref_code=[None],
            ref_spk_embedding=[spk_emb],
            x_vector_only_mode=[True],
            icl_mode=[False],
        )
        ref_ids = [None]

        # Insert for both append_silence variants to avoid cache misses
        for append_silence in (True, False):
            cache_key = (_CACHE_SENTINEL, "", True, append_silence)
            self._model._voice_prompt_cache[cache_key] = (vcp, ref_ids)

        if self.debug:
            print(f"{COLOR_GREEN}✓ Speaker embedding cached (voice: {voice.name}){COLOR_RESET}")

    def _warmup(self):
        """
        Pre‑captures CUDA graphs by running a dummy synthesis.
        Uses the sentinel key to hit the cached embedding.
        """
        if not self.current_voice:
            return

        if self._is_warmed_up:
            return

        if self.debug:
            print(f"{COLOR_YELLOW}🛠️  [FasterQwenEngine] Capturing CUDA Graphs (one‑time warmup)...{COLOR_RESET}", end="", flush=True)

        warmup_start = time.time()
        warmup_text = "Stabilizing the engine for real‑time generation."

        try:
            with torch.inference_mode():
                # Use the sentinel to trigger the cache
                generator = self._model.generate_voice_clone_streaming(
                    text=warmup_text,
                    language=self.current_voice.language,
                    ref_audio=_CACHE_SENTINEL,
                    ref_text="",
                    chunk_size=self.chunk_size,
                    xvec_only=True,
                    non_streaming_mode=False,
                    append_silence=True,
                    max_new_tokens=24  # enough to capture graphs
                )
                # Consume the generator
                for _ in generator:
                    pass

            if self.debug:
                print(f" {COLOR_GREEN}Done! ({time.time() - warmup_start:.2f}s){COLOR_RESET}")
        except Exception as e:
            if self.debug:
                print(f"\n{COLOR_RED}❌ Warmup failed: {e}{COLOR_RESET}")

        self._is_warmed_up = True


    def get_stream_info(self) -> Tuple[int, int, int]:
        """
        Returns PyAudio stream configuration.

        Returns:
            tuple: (format, channels, rate) -> 16-bit PCM, Mono, 24kHz.
        """
        return pyaudio.paInt16, 1, 24000

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text into audio data and yields it to the PyAudio queue.

        Args:
            text (str): The text to be converted to speech.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.current_voice:
            print(f"{COLOR_RED}❌ [FasterQwenEngine] No voice set. Please provide a FasterQwenVoice configuration.{COLOR_RESET}")
            return False

        if self.debug:
            print(f"\n{COLOR_CYAN}Synthesizing Text:{COLOR_RESET} {text}")
            print(f" ├─ {COLOR_YELLOW}Language:{COLOR_RESET} {self.current_voice.language}")

        start_time = time.time()
        first_chunk_received = False

        try:
            with torch.inference_mode():
                # Dynamically check what arguments are supported by the installed library
                sig = inspect.signature(self._model.generate_voice_clone_streaming)
                kwargs = {
                    "text": text,
                    "language": self.current_voice.language,
                    "ref_audio": _CACHE_SENTINEL,
                    "ref_text": "",
                    "chunk_size": self.chunk_size,
                    "xvec_only": self.xvec_only,
                    "non_streaming_mode": False,
                    "append_silence": True,    # recommended for clean starts
                }

                # Only inject 'instruct' if the installed version supports it
                if "instruct" in sig.parameters and self.current_voice.instruct:
                    kwargs["instruct"] = self.current_voice.instruct
                    if self.debug:
                        print(f" ├─ {COLOR_YELLOW}Instruct:{COLOR_RESET} {self.current_voice.instruct}")
                elif self.current_voice.instruct and "instruct" not in sig.parameters:
                    # Enhanced info when instruct is provided but not supported
                    if self.debug:
                        print(f" ├─ {COLOR_RED}⚠ Instruct parameter provided but NOT supported by this TTS version{COLOR_RESET}")
                        print(f" │   ├─ Instruct value: {COLOR_CYAN}{self.current_voice.instruct}{COLOR_RESET}")
                        print(f" │   ├─ Available parameters: {COLOR_GREEN}{', '.join(sig.parameters.keys())}{COLOR_RESET}")
                        print(f" │   └─ {COLOR_YELLOW}Instruction will be ignored{COLOR_RESET}")
                    else:
                        # Still log warning even if debug is off (optional)
                        print(f"{COLOR_RED}[WARNING] Instruct parameter '{self.current_voice.instruct}' provided but not supported by TTS version{COLOR_RESET}")

                # # Only inject 'instruct' if the installed version supports it
                # if "instruct" in sig.parameters and self.current_voice.instruct:
                #     kwargs["instruct"] = self.current_voice.instruct
                #     if self.debug:
                #         print(f" ├─ {COLOR_YELLOW}Instruct:{COLOR_RESET} {self.current_voice.instruct}")

                generator = self._model.generate_voice_clone_streaming(**kwargs)

                for chunk, sr, timing in generator:
                    if not first_chunk_received:
                        ttfa = (time.time() - start_time) * 1000
                        if self.debug:
                            print(f" ├─ {COLOR_GREEN}TTFA (Latency):{COLOR_RESET} {COLOR_BOLD}{ttfa:.2f}ms{COLOR_RESET} 🔈")
                            print(f" └─ {COLOR_CYAN}Streaming audio data...{COLOR_RESET}", end="\r")
                        first_chunk_received = True

                    # Chunk is already float32 in range [-1,1] (confirmed by library)
                    audio_data = chunk.astype(np.float32)

                    # Clamp to safe range (just in case)
                    audio_data = np.clip(audio_data, -1.0, 1.0)

                    # Convert to 16-bit PCM
                    audio_int16 = (audio_data * 32767).astype(np.int16).tobytes()
                    self.queue.put(audio_int16)

            if self.debug:
                print(f"\n {COLOR_GREEN}✓ Finished generation in {time.time() - start_time:.3f}s{COLOR_RESET}")

            return True

        except Exception as e:
            traceback.print_exc()
            print(f"\n{COLOR_RED}{COLOR_BOLD}❌ ERROR during synthesis:{COLOR_RESET} {e}")
            return False

    def get_voices(self) -> List[FasterQwenVoice]:
        """
        Returns the list of voices (currently empty as voices are dynamically defined).
        """
        return []

    def set_voice(self, voice: Union[str, FasterQwenVoice]):
        """
        Sets the voice configuration. If a speaker embedding file is provided,
        it is loaded once and cached. Otherwise, the reference audio/text is used
        to extract the embedding on‑the‑fly (also cached).

        Args:
            voice (Union[str, FasterQwenVoice]): A FasterQwenVoice object instance.
        """
        if isinstance(voice, FasterQwenVoice):
            self.current_voice = voice
        else:
            print(f"{COLOR_YELLOW}[FasterQwenEngine] Setting Voice via string is not supported. "
                  f"Please provide a FasterQwenVoice object.{COLOR_RESET}")
            return

        # Prime the cache with this voice's embedding
        self._prime_cache(self.current_voice)

        # Perform warmup (CUDA graph capture) after priming
        self._warmup()

    def set_voice_parameters(self, **voice_parameters):
        """
        Dynamically adjusts voice parameters (instruct, language, etc.).
        Changing the reference audio or speaker embedding would require a new voice.

        Args:
            **voice_parameters: Accepts "instruct", "language", "chunk_size", or "xvec_only".
        """
        if self.current_voice:
            if "instruct" in voice_parameters:
                self.current_voice.instruct = voice_parameters["instruct"]
            if "language" in voice_parameters:
                self.current_voice.language = voice_parameters["language"]

        if "chunk_size" in voice_parameters:
            self.chunk_size = voice_parameters["chunk_size"]
        if "xvec_only" in voice_parameters:
            self.xvec_only = voice_parameters["xvec_only"]

        if self.debug:
            print(f"{COLOR_BLUE}[FasterQwenEngine] Voice parameters updated.{COLOR_RESET}")

    def shutdown(self):
        """
        Shuts down the engine and performs cleanup.
        """
        if self.debug:
            print(f"{COLOR_BLUE}[FasterQwenEngine] Shutdown called.{COLOR_RESET}")
        # No explicit cleanup needed – CUDA graphs and model will be freed by GC.
        pass