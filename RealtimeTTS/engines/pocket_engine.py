"""
Pocket TTS Engine for RealtimeTTS

Requires:
- pip install pocket-tts torch

Pocket TTS is a lightweight 100M parameter TTS model from Kyutai Labs
that runs efficiently on CPU with ~6x real-time performance.

Features:
- CPU-optimized inference
- Voice cloning via WAV files
- ~200ms latency to first audio chunk
- English language only
"""

from .base_engine import BaseEngine
from queue import Queue
from typing import Union, Optional
import numpy as np
import traceback
import pyaudio
import time
import os


class PocketTTSVoice:
    """Represents a voice for the PocketTTS engine."""

    # Built-in voices available in pocket-tts
    BUILTIN_VOICES = [
        "alba", "marius", "javert", "jean",
        "fantine", "cosette", "eponine", "azelma"
    ]

    def __init__(
        self,
        name: str,
        audio_prompt_path: Optional[str] = None
    ):
        """
        Initialize a PocketTTS voice.

        Args:
            name: Voice identifier (built-in name or custom name for cloned voice)
            audio_prompt_path: Path to WAV file for voice cloning (optional)
        """
        self.name = name
        self.audio_prompt_path = audio_prompt_path
        self.is_builtin = name in self.BUILTIN_VOICES and audio_prompt_path is None

    def __repr__(self):
        if self.audio_prompt_path:
            return f"PocketTTSVoice(name='{self.name}', cloned_from='{self.audio_prompt_path}')"
        return f"PocketTTSVoice(name='{self.name}', builtin={self.is_builtin})"


class PocketTTSEngine(BaseEngine):
    """
    A text-to-speech engine using Kyutai Labs' Pocket TTS model.

    Pocket TTS is a lightweight 100M parameter model optimized for CPU inference,
    achieving ~6x real-time performance with ~200ms latency to first audio chunk.

    Example usage:
        engine = PocketTTSEngine(voice="alba")
        engine.synthesize("Hello, world!")

    For voice cloning:
        voice = PocketTTSVoice(name="custom", audio_prompt_path="path/to/voice.wav")
        engine = PocketTTSEngine(voice=voice)
    """

    def __init__(
        self,
        voice: Union[str, PocketTTSVoice] = "alba",
        trim_silence: bool = True,
        silence_threshold: float = 0.005,
        extra_start_ms: int = 15,
        extra_end_ms: int = 15,
        fade_in_ms: int = 10,
        fade_out_ms: int = 10,
        debug: bool = False
    ):
        """
        Initialize the PocketTTS engine.

        Args:
            voice: Voice to use (built-in name string or PocketTTSVoice object)
            trim_silence: Whether to trim silence from audio output
            silence_threshold: Threshold for silence detection
            extra_start_ms: Extra milliseconds to trim from start
            extra_end_ms: Extra milliseconds to trim from end
            fade_in_ms: Fade-in duration in milliseconds
            fade_out_ms: Fade-out duration in milliseconds
            debug: Enable debug output
        """
        super().__init__()
        self.engine_name = "pocket_tts"
        self.debug = debug
        self.queue = Queue()

        # Silence trimming settings
        self.trim_silence = trim_silence
        self.silence_threshold = silence_threshold
        self.extra_start_ms = extra_start_ms
        self.extra_end_ms = extra_end_ms
        self.fade_in_ms = fade_in_ms
        self.fade_out_ms = fade_out_ms

        # Initialize the model
        self.model = None
        self.sample_rate = None
        self._voice_states = {}  # Cache for voice states
        self.current_voice = None
        self.current_voice_state = None

        self._load_model()
        self.set_voice(voice)

        if self.debug:
            print(f"[PocketTTSEngine] Initialized with voice: {self.current_voice}")

    def _load_model(self):
        """Load the Pocket TTS model."""
        try:
            from pocket_tts import TTSModel

            if self.debug:
                print("[PocketTTSEngine] Loading Pocket TTS model...")

            self.model = TTSModel.load_model()
            self.sample_rate = self.model.sample_rate

            if self.debug:
                print(f"[PocketTTSEngine] Model loaded. Sample rate: {self.sample_rate}")

        except ImportError:
            raise ImportError(
                "pocket-tts is not installed. Install it with: pip install pocket-tts"
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to load Pocket TTS model: {e}")

    def _get_voice_state(self, voice: Union[str, PocketTTSVoice]):
        """
        Get or create a voice state for the given voice.

        Args:
            voice: Voice identifier or PocketTTSVoice object

        Returns:
            Voice state object for use with generate_audio
        """
        if isinstance(voice, PocketTTSVoice):
            voice_name = voice.name
            audio_path = voice.audio_prompt_path
        else:
            voice_name = voice
            audio_path = None

        # Check cache
        cache_key = f"{voice_name}:{audio_path or 'builtin'}"
        if cache_key in self._voice_states:
            if self.debug:
                print(f"[PocketTTSEngine] Using cached voice state for: {cache_key}")
            return self._voice_states[cache_key]

        # Create new voice state
        if audio_path:
            # Voice cloning from WAV file
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Voice audio file not found: {audio_path}")

            if self.debug:
                print(f"[PocketTTSEngine] Creating voice state from audio: {audio_path}")

            voice_state = self.model.get_state_for_audio_prompt(audio_path)
        else:
            # Built-in voice
            if voice_name not in PocketTTSVoice.BUILTIN_VOICES:
                raise ValueError(
                    f"Unknown voice: {voice_name}. "
                    f"Available voices: {PocketTTSVoice.BUILTIN_VOICES}"
                )

            if self.debug:
                print(f"[PocketTTSEngine] Creating voice state for built-in voice: {voice_name}")

            voice_state = self.model.get_state_for_audio_prompt(voice_name)

        # Cache and return
        self._voice_states[cache_key] = voice_state
        return voice_state

    def get_stream_info(self):
        """
        Returns the audio stream configuration for PyAudio.

        Returns:
            tuple: (format, channels, sample_rate)
        """
        # Pocket TTS typically uses 24kHz sample rate
        sample_rate = self.sample_rate if self.sample_rate else 24000
        return (pyaudio.paInt16, 1, sample_rate)

    def synthesize(self, text: str) -> bool:
        """
        Synthesize text to speech audio.

        Args:
            text: Text to synthesize

        Returns:
            bool: True if synthesis was successful, False otherwise
        """
        if not text or not text.strip():
            if self.debug:
                print("[PocketTTSEngine] Empty text, skipping synthesis")
            return True

        start_time = time.time()

        try:
            if self.model is None:
                print("[PocketTTSEngine] Model not loaded")
                return False

            if self.current_voice_state is None:
                print("[PocketTTSEngine] No voice set")
                return False

            if self.debug:
                print(f"[PocketTTSEngine] Synthesizing: '{text[:50]}...'")

            # Generate audio
            audio_tensor = self.model.generate_audio(self.current_voice_state, text)

            # Convert to numpy array
            audio_float32 = audio_tensor.numpy()

            # Ensure 1D array
            if audio_float32.ndim > 1:
                audio_float32 = audio_float32.squeeze()

            # Trim silence if enabled
            if self.trim_silence:
                audio_float32 = self._trim_silence(
                    audio_float32,
                    sample_rate=self.sample_rate,
                    silence_threshold=self.silence_threshold,
                    extra_start_ms=self.extra_start_ms,
                    extra_end_ms=self.extra_end_ms,
                    fade_in_ms=self.fade_in_ms,
                    fade_out_ms=self.fade_out_ms,
                )

            # Convert to int16 bytes
            audio_int16 = (audio_float32 * 32767).astype(np.int16).tobytes()

            # Update audio duration tracking
            audio_length_seconds = len(audio_float32) / self.sample_rate
            self.audio_duration += audio_length_seconds

            # Put audio in queue
            self.queue.put(audio_int16)

            if self.debug:
                duration = time.time() - start_time
                print(f"[PocketTTSEngine] Synthesis completed in {duration:.3f}s "
                      f"({audio_length_seconds:.2f}s of audio)")

            return True

        except Exception as e:
            traceback.print_exc()
            print(f"[PocketTTSEngine] Error during synthesis: {e}")
            return False

    def get_voices(self) -> list:
        """
        Get list of available voices.

        Returns:
            list: List of PocketTTSVoice objects for built-in voices
        """
        return [PocketTTSVoice(name) for name in PocketTTSVoice.BUILTIN_VOICES]

    def set_voice(self, voice: Union[str, PocketTTSVoice]):
        """
        Set the current voice for synthesis.

        Args:
            voice: Voice identifier (string) or PocketTTSVoice object
        """
        try:
            if isinstance(voice, str):
                # Check if it's a built-in voice
                if voice in PocketTTSVoice.BUILTIN_VOICES:
                    self.current_voice = PocketTTSVoice(voice)
                else:
                    # Assume it's a path to a WAV file for cloning
                    if os.path.exists(voice):
                        self.current_voice = PocketTTSVoice(
                            name=os.path.basename(voice),
                            audio_prompt_path=voice
                        )
                    else:
                        raise ValueError(
                            f"Unknown voice: {voice}. "
                            f"Available voices: {PocketTTSVoice.BUILTIN_VOICES}"
                        )
            else:
                self.current_voice = voice

            # Get/create voice state
            self.current_voice_state = self._get_voice_state(self.current_voice)

            if self.debug:
                print(f"[PocketTTSEngine] Voice set to: {self.current_voice}")

        except Exception as e:
            traceback.print_exc()
            print(f"[PocketTTSEngine] Error setting voice: {e}")
            raise

    def set_voice_parameters(self, **voice_parameters):
        """
        Set voice parameters.

        Currently Pocket TTS doesn't support runtime voice parameters
        like speed adjustment.

        Args:
            **voice_parameters: Voice parameters (currently unused)
        """
        if self.debug:
            print(f"[PocketTTSEngine] set_voice_parameters called with: {voice_parameters}")
            print("[PocketTTSEngine] Note: Pocket TTS doesn't support runtime voice parameters")

    def shutdown(self):
        """Shutdown the engine and release resources."""
        if self.debug:
            print("[PocketTTSEngine] Shutdown called")

        # Clear caches
        self._voice_states.clear()
        self.current_voice_state = None
        self.model = None
