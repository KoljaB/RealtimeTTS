"""
NeuTTS Engine for RealtimeTTS.

NeuTTS is an open-source on-device TTS with voice cloning capabilities.
https://github.com/neuphonic/neutts
"""

import logging
import os
import sys
from typing import Union, Optional

import pyaudio

from .base_engine import BaseEngine

# Add NeuTTS to path if installed as a git clone (not a package)
_neutts_paths = [
    os.path.expanduser("~/Desktop/TTS/neutts"),
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "neutts"),
]
for _path in _neutts_paths:
    if os.path.exists(_path) and _path not in sys.path:
        sys.path.insert(0, _path)
        break


class NeuTTSVoice:
    """Represents a NeuTTS voice (reference audio for cloning)."""

    def __init__(self, name: str, ref_audio_path: str, ref_text: str, ref_codes=None):
        self.name = name
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.ref_codes = ref_codes  # Pre-encoded reference codes

    def __repr__(self):
        return f"NeuTTSVoice({self.name}, audio={self.ref_audio_path})"


class NeuTTSEngine(BaseEngine):
    """
    NeuTTS Engine for on-device text-to-speech with voice cloning.

    NeuTTS supports instant voice cloning with as little as 3 seconds of audio.
    """

    def __init__(
        self,
        model: str = "neutts-nano",
        backbone_repo: str = "neuphonic/neutts-nano",
        codec_repo: str = "neuphonic/neucodec",
        device: str = "cpu",
        voices_dir: Optional[str] = None,
        default_voice: Optional[str] = None,
    ):
        """
        Initialize the NeuTTS engine.

        Args:
            model: Model variant to use ("neutts-nano" or "neutts-air")
            backbone_repo: HuggingFace repo for the backbone model
            codec_repo: HuggingFace repo for the codec model
            device: Device to run inference on ("cpu", "cuda", "mps")
            voices_dir: Directory containing voice reference files
            default_voice: Name of the default voice to use
        """
        super().__init__()
        self.engine_name = "neutts"
        self.model_name = model
        self.backbone_repo = backbone_repo
        self.codec_repo = codec_repo
        self.device = device
        self.voices_dir = voices_dir or os.path.join(os.path.dirname(__file__), "voices")
        self.default_voice = default_voice

        self._tts = None
        self._voices: dict[str, NeuTTSVoice] = {}
        self._current_voice: Optional[NeuTTSVoice] = None

        self._init_model()
        self._load_voices()

    def _init_model(self):
        """Initialize the NeuTTS model."""
        try:
            from neutts import NeuTTS

            logging.info(f"Initializing NeuTTS with {self.backbone_repo}")
            self._tts = NeuTTS(
                backbone_repo=self.backbone_repo,
                backbone_device=self.device,
                codec_repo=self.codec_repo,
                codec_device=self.device,
            )
            logging.info("NeuTTS model initialized successfully")
        except ImportError:
            raise ImportError(
                "NeuTTS is not installed. Install with:\n"
                "  git clone https://github.com/neuphonic/neutts.git\n"
                "  cd neutts && pip install -r requirements.txt"
            )
        except Exception as e:
            logging.error(f"Failed to initialize NeuTTS: {e}")
            raise

    def _load_voices(self):
        """Load voice references from the voices directory."""
        if not os.path.exists(self.voices_dir):
            os.makedirs(self.voices_dir, exist_ok=True)
            logging.info(f"Created voices directory: {self.voices_dir}")
            return

        # Look for .wav files with matching .txt files
        for filename in os.listdir(self.voices_dir):
            if filename.endswith(".wav"):
                voice_name = os.path.splitext(filename)[0]
                audio_path = os.path.join(self.voices_dir, filename)
                text_path = os.path.join(self.voices_dir, f"{voice_name}.txt")

                if os.path.exists(text_path):
                    with open(text_path, "r") as f:
                        ref_text = f.read().strip()

                    # Pre-encode reference codes
                    try:
                        ref_codes = self._tts.encode_reference(audio_path)
                        voice = NeuTTSVoice(
                            name=voice_name,
                            ref_audio_path=audio_path,
                            ref_text=ref_text,
                            ref_codes=ref_codes,
                        )
                        self._voices[voice_name] = voice
                        logging.info(f"Loaded voice: {voice_name}")
                    except Exception as e:
                        logging.warning(f"Failed to load voice {voice_name}: {e}")
                else:
                    logging.warning(
                        f"Voice {voice_name} missing transcript file: {text_path}"
                    )

        # Set default voice
        if self._voices:
            if self.default_voice and self.default_voice in self._voices:
                self._current_voice = self._voices[self.default_voice]
            else:
                self._current_voice = list(self._voices.values())[0]
            logging.info(f"Default voice set to: {self._current_voice.name}")

    def get_stream_info(self):
        """
        Returns the audio stream configuration for PyAudio.

        NeuTTS outputs 24kHz mono 16-bit PCM audio.

        Returns:
            tuple: (format, channels, sample_rate)
        """
        return pyaudio.paInt16, 1, 24000

    def synthesize(self, text: str) -> bool:
        """
        Synthesize text to audio using voice cloning.

        Args:
            text: Text to synthesize

        Returns:
            True if synthesis succeeded, False otherwise
        """
        super().synthesize(text)

        if not self._current_voice:
            logging.error("No voice selected for NeuTTS synthesis")
            return False

        try:
            logging.debug(f"Synthesizing with NeuTTS: {text[:50]}...")

            # Generate audio using the current voice reference
            wav = self._tts.infer(
                text,
                self._current_voice.ref_codes,
                self._current_voice.ref_text,
            )

            # Convert float32 numpy array to int16 bytes
            # NeuTTS returns float32 audio normalized to [-1, 1]
            import numpy as np

            audio_int16 = (wav * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            # Send audio in chunks
            chunk_size = 4096
            for i in range(0, len(audio_bytes), chunk_size):
                if self.stop_synthesis_event.is_set():
                    logging.debug("Synthesis stopped by event")
                    break
                chunk = audio_bytes[i : i + chunk_size]
                self.queue.put(chunk)

            return True

        except Exception as e:
            logging.error(f"NeuTTS synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_voices(self) -> list:
        """
        Get list of available voices.

        Returns:
            List of NeuTTSVoice objects
        """
        return list(self._voices.values())

    def set_voice(self, voice: Union[str, NeuTTSVoice]):
        """
        Set the voice to use for synthesis.

        Args:
            voice: Voice name (str) or NeuTTSVoice object
        """
        if isinstance(voice, NeuTTSVoice):
            self._current_voice = voice
            logging.info(f"Voice set to: {voice.name}")
            return

        # Find voice by name
        if voice in self._voices:
            self._current_voice = self._voices[voice]
            logging.info(f"Voice set to: {voice}")
            return

        # Partial match
        for name, v in self._voices.items():
            if voice.lower() in name.lower():
                self._current_voice = v
                logging.info(f"Voice set to: {name}")
                return

        logging.warning(f"Voice '{voice}' not found")

    def set_voice_parameters(self, **voice_parameters):
        """
        Set voice parameters.

        Currently NeuTTS doesn't support runtime voice parameters,
        but this method is required by BaseEngine.
        """
        pass

    def add_voice(
        self,
        name: str,
        audio_path: str,
        transcript: str,
    ) -> NeuTTSVoice:
        """
        Add a new voice from reference audio.

        Args:
            name: Name for this voice
            audio_path: Path to reference audio file (3-15 seconds, clean speech)
            transcript: Exact transcript of the reference audio

        Returns:
            The created NeuTTSVoice object
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            ref_codes = self._tts.encode_reference(audio_path)
            voice = NeuTTSVoice(
                name=name,
                ref_audio_path=audio_path,
                ref_text=transcript,
                ref_codes=ref_codes,
            )
            self._voices[name] = voice

            if self._current_voice is None:
                self._current_voice = voice

            logging.info(f"Added voice: {name}")
            return voice

        except Exception as e:
            logging.error(f"Failed to add voice {name}: {e}")
            raise

    def clone_voice(self, audio_path: str, transcript: str) -> NeuTTSVoice:
        """
        Convenience method to clone a voice temporarily without saving.

        Args:
            audio_path: Path to reference audio
            transcript: Transcript of the reference audio

        Returns:
            NeuTTSVoice object (not saved to voices list)
        """
        ref_codes = self._tts.encode_reference(audio_path)
        return NeuTTSVoice(
            name="cloned",
            ref_audio_path=audio_path,
            ref_text=transcript,
            ref_codes=ref_codes,
        )

    def shutdown(self):
        """Clean up resources."""
        self._tts = None
        self._voices.clear()
        self._current_voice = None
