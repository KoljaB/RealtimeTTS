import os
import time
import traceback
import inspect  # <-- Added to safely check function arguments
import pyaudio
import numpy as np
from queue import Queue
from typing import Optional, Union, List

try:
    import torch
    from faster_qwen3_tts import FasterQwen3TTS
except ImportError:
    torch = None
    FasterQwen3TTS = None

from .base_engine import BaseEngine

# --- ANSI escape codes for debug styling ---
COLOR_BLUE   = "\033[94m"
COLOR_GREEN  = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN   = "\033[96m"
COLOR_RED    = "\033[91m"
COLOR_BOLD   = "\033[1m"
COLOR_RESET  = "\033[0m"


class FasterQwenVoice:
    """
    Represents a FasterQwenTTS voice configuration for voice cloning and instruction.

    Args:
        name (str): Identifier for this voice.
        ref_audio (str): Path to the reference audio file (.wav).
        ref_text (str): The exact transcription of the reference audio.
        language (str): Target language for synthesis (e.g., "English", "German").
        instruct (str): Instruction or emotion prompt (e.g., "Speak with a happy voice.").
    """
    def __init__(
            self,
            name: str,
            ref_audio: str,
            ref_text: str,
            language: str = "English",
            instruct: str = "Speak normally."
    ):
        self.name = name
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.language = language
        self.instruct = instruct

    def __repr__(self):
        return f"FasterQwenVoice(name={self.name}, lang={self.language}, instruct={self.instruct})"


class FasterQwenEngine(BaseEngine):
    """
    A real-time text-to-speech engine that uses the faster-qwen3-tts library.
    It leverages CUDA graph capture and static KV caching for ultra-low latency.
    """

    def __init__(
            self,
            model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device: str = "cuda",
            voice: Optional[Union[str, FasterQwenVoice]] = None,
            chunk_size: int = 8,
            xvec_only: bool = True,
            debug: bool = False
    ):
        """
        Initializes the FasterQwenTTS engine.

        Args:
            model_name (str): The HF repo ID for the model (e.g. "Qwen/Qwen3-TTS-12Hz-0.6B-Base").
            device (str): Device to run the model on (defaults to "cuda").
            voice (Optional[FasterQwenVoice]): The voice configuration to use.
            chunk_size (int): Number of steps per chunk yielded (8 steps ≈ 667ms of audio). 
                              Lower = faster initial TTFA but more overhead. Default is 8.
            xvec_only (bool): If True, uses only speaker embedding (cleaner lang switching, lower latency).
                              If False, uses full In-Context Learning. Default is True.
            debug (bool): If True, prints verbose streaming and latency metrics.
        """
        super().__init__()

        if FasterQwen3TTS is None or torch is None:
            raise ImportError(
                "faster-qwen3-tts or torch is not installed. "
                "Please install them via: pip install faster-qwen3-tts torch"
            )

        self.model_name = model_name
        self.device = device
        self.chunk_size = chunk_size
        self.xvec_only = xvec_only
        self.debug = debug
        
        self.queue = Queue()
        self.current_voice = None

        if self.debug:
            print(f"\n{COLOR_CYAN}🧠 [FasterQwenEngine] Loading Model ({self.model_name}) on {self.device}...{COLOR_RESET}")
        
        load_start = time.time()
        self.model = FasterQwen3TTS.from_pretrained(self.model_name, device=self.device, dtype=torch.float32)
        
        if self.debug:
            print(f"{COLOR_GREEN}✓ Model loaded in {time.time() - load_start:.2f}s{COLOR_RESET}")

        if voice is not None:
            self.set_voice(voice)
            self._warmup()

        self.post_init()

    def post_init(self):
        self.engine_name = "faster_qwen3"

    def _warmup(self):
        """
        Pre-captures CUDA graphs by running a dummy stream. Ensures the first
        real-time synthesis isn't lagging due to graph compilation.
        """
        if not self.current_voice or not os.path.exists(self.current_voice.ref_audio):
            if self.debug:
                print(f"{COLOR_YELLOW}⚠️ [FasterQwenEngine] Ref audio missing. Skipping warmup.{COLOR_RESET}")
            return

        if self.debug:
            print(f"{COLOR_YELLOW}🛠️  [FasterQwenEngine] Capturing CUDA Graphs (One-time warmup)...{COLOR_RESET}", end="", flush=True)
        
        warmup_start = time.time()
        warmup_text = "Stabilizing the engine for real-time generation."

        try:
            with torch.inference_mode():
                # Dynamically check what arguments are supported by the installed library version
                sig = inspect.signature(self.model.generate_voice_clone_streaming)
                kwargs = {
                    "text": warmup_text,
                    "language": self.current_voice.language,
                    "ref_audio": self.current_voice.ref_audio,
                    "ref_text": self.current_voice.ref_text,
                    "chunk_size": self.chunk_size,
                    "xvec_only": self.xvec_only
                }

                # Only inject 'instruct' if the installed version supports it
                if "instruct" in sig.parameters:
                    if self.current_voice.instruct:
                        kwargs["instruct"] = self.current_voice.instruct
                
                list(self.model.generate_voice_clone_streaming(**kwargs))
                
            if self.debug:
                print(f" {COLOR_GREEN}Done! ({time.time() - warmup_start:.2f}s){COLOR_RESET}")
        except Exception as e:
            if self.debug:
                print(f"\n{COLOR_RED}❌ Warmup failed: {e}{COLOR_RESET}")

    def get_stream_info(self):
        """
        Returns PyAudio stream configuration for FasterQwenTTS.

        Returns:
            tuple: (format, channels, rate) -> 16-bit PCM, Mono, 24kHz.
        """
        return pyaudio.paInt16, 1, 24000

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text into audio data and yields it directly to the PyAudio queue.

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
                # Dynamically check what arguments are supported
                sig = inspect.signature(self.model.generate_voice_clone_streaming)
                kwargs = {
                    "text": text,
                    "language": self.current_voice.language,
                    "ref_audio": self.current_voice.ref_audio,
                    "ref_text": self.current_voice.ref_text,
                    "chunk_size": self.chunk_size,
                    "xvec_only": self.xvec_only
                }

                # Only inject 'instruct' if the installed version supports it
                if "instruct" in sig.parameters:
                    if self.current_voice.instruct:
                        kwargs["instruct"] = self.current_voice.instruct
                        if self.debug:
                            print(f" ├─ {COLOR_YELLOW}Instruct:{COLOR_RESET} {self.current_voice.instruct}")
                else:
                    if self.debug and self.current_voice.instruct:
                        print(f" ├─ {COLOR_YELLOW}Instruct:{COLOR_RESET} ⚠️ Ignored ('instruct' not supported in your faster-qwen3-tts version)")

                generator = self.model.generate_voice_clone_streaming(**kwargs)

                for chunk, sr, timing in generator:
                    if not first_chunk_received:
                        ttfa = (time.time() - start_time) * 1000
                        if self.debug:
                            print(f" ├─ {COLOR_GREEN}TTFA (Latency):{COLOR_RESET} {COLOR_BOLD}{ttfa:.2f}ms{COLOR_RESET} 🔈")
                            print(f" └─ {COLOR_CYAN}Streaming audio data...{COLOR_RESET}", end="\r")
                        first_chunk_received = True

                    # Chunk preprocessing and normalization
                    audio_data = chunk.astype(np.float32)
                    
                    # Ensure values are strictly between -1.0 and 1.0 
                    if np.max(np.abs(audio_data)) > 1.1:
                        audio_data = audio_data / 32768.0

                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    
                    # Convert to PyAudio compatible 16-bit PCM bytes
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
        FasterQwen-TTS defines voices dynamically using cloned references.
        Returning an empty list as standard predefined voices aren't strictly hardcoded.
        """
        return[]

    def set_voice(self, voice: Union[str, FasterQwenVoice]):
        """
        Sets the Voice configuration (Reference Audio + Text + Language + Instruct).

        Args:
            voice (Union[str, FasterQwenVoice]): A FasterQwenVoice object instance.
        """
        if isinstance(voice, FasterQwenVoice):
            self.current_voice = voice
        else:
            print(f"{COLOR_YELLOW}[FasterQwenEngine] Setting Voice via string is not recommended. "
                  f"Please provide a FasterQwenVoice object for proper voice cloning.{COLOR_RESET}")

    def set_voice_parameters(self, **voice_parameters):
        """
        Dynamically adjusts Voice parameters (useful for shifting emotions on the fly).
        
        Args:
            **voice_parameters: Accepts "instruct", "language", "ref_audio", "ref_text", 
                                "chunk_size", or "xvec_only".
        """
        if self.current_voice:
            if "instruct" in voice_parameters:
                self.current_voice.instruct = voice_parameters["instruct"]
            if "language" in voice_parameters:
                self.current_voice.language = voice_parameters["language"]
            if "ref_audio" in voice_parameters:
                self.current_voice.ref_audio = voice_parameters["ref_audio"]
            if "ref_text" in voice_parameters:
                self.current_voice.ref_text = voice_parameters["ref_text"]

        if "chunk_size" in voice_parameters:
            self.chunk_size = voice_parameters["chunk_size"]
        if "xvec_only" in voice_parameters:
            self.xvec_only = voice_parameters["xvec_only"]
        
        if self.debug:
            print(f"{COLOR_BLUE}[FasterQwenEngine] Voice parameters updated.{COLOR_RESET}")

    def shutdown(self):
        """
        Shuts down the Engine and performs cleanup if necessary.
        """
        if self.debug:
            print(f"{COLOR_BLUE}[FasterQwenEngine] Shutdown called.{COLOR_RESET}")
        pass
