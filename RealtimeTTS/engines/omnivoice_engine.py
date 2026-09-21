import os
import time
import traceback
import pyaudio
import numpy as np
from typing import Optional, Union, List, Tuple

try:
    import torch
    from omnivoice import OmniVoice
except ImportError:
    torch = None
    OmniVoice = None

from .base_engine import BaseEngine

# --- ANSI escape codes for debug styling ---
COLOR_BLUE   = "\033[94m"
COLOR_GREEN  = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN   = "\033[96m"
COLOR_RED    = "\033[91m"
COLOR_BOLD   = "\033[1m"
COLOR_RESET  = "\033[0m"


class OmniVoiceVoice:
    """
    Represents a voice for the OmniVoiceEngine.

    Stores the reference audio and transcription required for OmniVoice's
    zero-shot voice cloning capabilities.
    """
    def __init__(
            self,
            name: str,
            ref_audio: str,
            ref_text: str,
            language: str = "en"
    ):
        self.name = name
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.language = language

        if not ref_audio or not ref_text:
            raise ValueError(
                "OmniVoiceVoice requires both ref_audio and ref_text for voice cloning."
            )

    def __repr__(self):
        return (
            f"OmniVoiceVoice(name={self.name}, lang={self.language}, "
            f"ref_audio={self.ref_audio})"
        )


class OmniVoiceEngine(BaseEngine):
    """
    Text-to-Speech engine using k2-fsa/OmniVoice for RealtimeTTS.
    """

    def __init__(
            self,
            model_id: str = "k2-fsa/OmniVoice",
            device_map: str = "cuda:0",
            dtype=None,
            voice: Optional[Union[str, OmniVoiceVoice]] = None,
            num_steps_schedule: Optional[List[int]] = None,
            preprocess_prompt: bool = True,
            postprocess_output: bool = True,
            debug: bool = False
    ):
        """
        Initializes the OmniVoiceTTS engine.

        Args:
            model_id (str): HF repo ID for the model.
            device_map (str): Device to run the model on.
            dtype (torch.dtype): Data type for the model (defaults to float16).
            voice (Optional[OmniVoiceVoice]): The voice configuration containing cloning parameters.
            num_steps_schedule (List[int]): Per-sentence generation steps.
                Example: [12, 32] means first sentence uses 12 steps,
                second and all following sentences use 32 steps.
                Last value is reused for all further sentences.
            preprocess_prompt (bool): If True, preprocesses the input text before generation.
            postprocess_output (bool): If True, postprocesses the generated audio output.
            debug (bool): If True, prints verbose processing metrics.
        """
        super().__init__()

        if OmniVoice is None or torch is None:
            raise ImportError(
                "OmniVoice or torch is not installed. "
                "Please install them via: pip install torch torchaudio omnivoice"
            )

        if voice is None:
            raise ValueError(
                "OmniVoiceEngine requires a voice configuration. "
                "Please provide an OmniVoiceVoice object with reference audio and text."
            )

        if dtype is None:
            dtype = torch.float16

        if num_steps_schedule is None:
            num_steps_schedule = [12, 32]

        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.num_steps_schedule = self._normalize_num_steps_schedule(num_steps_schedule)
        self.debug = debug
        self.preprocess_prompt = preprocess_prompt
        self.postprocess_output = postprocess_output

        self.current_voice = None
        self._model = None
        self._is_warmed_up = False

        if self.debug:
            print(
                f"\n{COLOR_CYAN}🧠 [OmniVoiceEngine] Loading Model "
                f"({self.model_id}) on {self.device_map}...{COLOR_RESET}"
            )

        load_start = time.time()
        self._model = OmniVoice.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            dtype=self.dtype
        )
        if self.debug:
            print(f"{COLOR_GREEN}✓ Model loaded in {time.time() - load_start:.2f}s{COLOR_RESET}")

        self.set_voice(voice)
        self._warmup()

        self.post_init()

    def post_init(self):
        self.engine_name = "omnivoice"

    def _normalize_num_steps_schedule(self, schedule: List[int]) -> List[int]:
        """
        Validates and normalizes the step schedule.
        """
        if not isinstance(schedule, list) or len(schedule) == 0:
            raise ValueError("num_steps_schedule must be a non-empty list of positive integers.")

        normalized = []
        for value in schedule:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    "num_steps_schedule must contain only positive integers."
                )
            normalized.append(value)

        return normalized

    def _get_num_steps_for_sentence(self, sentence_count: int) -> int:
        """
        Returns the configured number of steps for the given sentence index.

        Behavior:
        - first sentence -> first entry
        - second sentence -> second entry
        - ...
        - once the list is exhausted, reuse the last entry
        """
        if sentence_count <= 1:
            index = 0
        else:
            index = sentence_count - 1

        index = min(index, len(self.num_steps_schedule) - 1)
        return self.num_steps_schedule[index]

    def _warmup(self):
        """
        Performs a warm-up generation run to initialize GPU caches.
        """
        if not self.current_voice or self._is_warmed_up:
            return

        if self.debug:
            print(
                f"{COLOR_YELLOW}🛠️  [OmniVoiceEngine] Performing warm-up run..."
                f"{COLOR_RESET}",
                end="",
                flush=True
            )

        warmup_start = time.time()
        try:
            with torch.no_grad():
                _ = self._model.generate(
                    language=self.current_voice.language,
                    text="Warmup sequence.",
                    ref_audio=self.current_voice.ref_audio,
                    ref_text=self.current_voice.ref_text,
                    num_step=self.num_steps_schedule[0],
                    preprocess_prompt=self.preprocess_prompt,
                    postprocess_output=self.postprocess_output
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if self.debug:
                print(f" {COLOR_GREEN}Done! ({time.time() - warmup_start:.2f}s){COLOR_RESET}")
        except Exception as e:
            if self.debug:
                print(f"\n{COLOR_RED}❌ Warmup failed: {e}{COLOR_RESET}")

        self._is_warmed_up = True

    def get_stream_info(self) -> Tuple[int, int, int]:
        """
        Returns PyAudio stream configuration.
        OmniVoice generates audio at 24000 Hz.
        """
        return pyaudio.paInt16, 1, 24000

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
            sentence_count (int): Sentence index tracker.
        """
        super().synthesize(text, sentence_count)

        if not self.current_voice:
            print(
                f"{COLOR_RED}❌ [OmniVoiceEngine] No voice set. "
                f"Please provide an OmniVoiceVoice configuration.{COLOR_RESET}"
            )
            return False

        current_num_steps = self._get_num_steps_for_sentence(sentence_count)

        if self.debug:
            print(f"\n{COLOR_CYAN}Synthesizing Text:{COLOR_RESET} {text}")
            print(f" ├─ {COLOR_YELLOW}Voice:{COLOR_RESET} {self.current_voice.name}")
            print(f" ├─ {COLOR_YELLOW}Language:{COLOR_RESET} {self.current_voice.language}")
            print(f" ├─ {COLOR_YELLOW}Steps:{COLOR_RESET} {current_num_steps}")
            print(f" ├─ {COLOR_YELLOW}Schedule:{COLOR_RESET} {self.num_steps_schedule}")

        start_time = time.time()

        try:
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                generate_start = time.perf_counter()
                audio = self._model.generate(
                    language=self.current_voice.language,
                    text=text,
                    ref_audio=self.current_voice.ref_audio,
                    ref_text=self.current_voice.ref_text,
                    num_step=current_num_steps,
                    preprocess_prompt=self.preprocess_prompt,
                    postprocess_output=self.postprocess_output
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                generate_end = time.perf_counter()

            # The audio tensor shape is typically [1, samples] or [samples]
            if hasattr(audio[0], 'cpu'):
                waveform = audio[0].cpu().numpy()
            else:
                waveform = audio[0]
            if waveform.ndim > 1:
                waveform = waveform[0]  # Ensure mono

            # Clip limits to prevent clipping artifacts and convert to 16-bit PCM
            waveform = np.clip(waveform, -1.0, 1.0)
            audio_int16 = (waveform * 32767).astype(np.int16).tobytes()

            # --- Calculate Metrics (RTF) ---
            processing_time = generate_end - generate_start
            num_samples = waveform.shape[-1]
            sample_rate = 24000
            audio_duration = num_samples / sample_rate

            rtf = processing_time / audio_duration if audio_duration > 0 else 0.0
            speed_factor = 1 / rtf if rtf > 0 else 0.0
            ttfa = (time.time() - start_time) * 1000

            # Queue the entire audio chunk for the RealtimeTTS player
            self.queue.put(audio_int16)

            if self.debug:
                print(f" ├─ {COLOR_GREEN}TTFA (Latency):{COLOR_RESET} {COLOR_BOLD}{ttfa:.2f}ms{COLOR_RESET} 🔈")
                print(f" ├─ {COLOR_YELLOW}Processing Time:{COLOR_RESET} {processing_time:.4f}s")
                print(f" ├─ {COLOR_YELLOW}Audio Duration:{COLOR_RESET} {audio_duration:.4f}s")
                print(f" ├─ {COLOR_GREEN}RTF:{COLOR_RESET} {COLOR_BOLD}{rtf:.4f}{COLOR_RESET} ({speed_factor:.2f}x Realtime)")
                print(f" └─ {COLOR_GREEN}✓ Finished generation{COLOR_RESET}")

            return True

        except Exception as e:
            traceback.print_exc()
            print(f"\n{COLOR_RED}{COLOR_BOLD}❌ ERROR during synthesis:{COLOR_RESET} {e}")
            return False

    def get_voices(self) -> List[OmniVoiceVoice]:
        """
        Returns an empty list as voices are user-defined via paths.
        """
        return []

    def set_voice(self, voice: Union[str, OmniVoiceVoice]):
        """
        Sets the voice configuration.
        """
        if isinstance(voice, OmniVoiceVoice):
            self.current_voice = voice
        else:
            print(
                f"{COLOR_YELLOW}[OmniVoiceEngine] Setting Voice via string is not supported. "
                f"Please provide an OmniVoiceVoice object.{COLOR_RESET}"
            )

    def set_voice_parameters(self, **voice_parameters):
        """
        Dynamically adjusts voice parameters.
        """
        if self.current_voice:
            if "language" in voice_parameters:
                self.current_voice.language = voice_parameters["language"]

        if "num_steps_schedule" in voice_parameters:
            self.num_steps_schedule = self._normalize_num_steps_schedule(
                voice_parameters["num_steps_schedule"]
            )

        if self.debug:
            print(f"{COLOR_BLUE}[OmniVoiceEngine] Voice parameters updated.{COLOR_RESET}")

    def shutdown(self):
        """
        Shuts down the engine and releases model memory.
        """
        if self.debug:
            print(f"{COLOR_BLUE}[OmniVoiceEngine] Shutdown called.{COLOR_RESET}")
        self._model = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
