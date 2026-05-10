"""
NeuTTS Engine for RealtimeTTS.

NeuTTS is an open-source on-device TTS with voice cloning capabilities.
https://github.com/neuphonic/neutts
"""

import logging
import os
import sys
from typing import Optional, Union

from .base_engine import BaseEngine

# Add NeuTTS to path if installed as a git clone (not a package).
_neutts_paths = [
    os.path.expanduser("~/Desktop/TTS/neutts"),
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "neutts"),
]
for _path in _neutts_paths:
    if os.path.exists(_path) and _path not in sys.path:
        sys.path.insert(0, _path)
        break


class NeuTTSVoice:
    """Reference audio configuration for NeuTTS voice cloning."""

    def __init__(self, name: str, ref_audio_path: str, ref_text: str, ref_codes=None):
        if not os.path.exists(ref_audio_path):
            raise FileNotFoundError(
                "NeuTTS reference WAV file not found at: %s" % ref_audio_path
            )
        if not ref_text or not ref_text.strip():
            raise ValueError("NeuTTSVoice requires the exact reference transcript.")
        self.name = name
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text.strip()
        self.ref_codes = ref_codes

    def __repr__(self):
        return "NeuTTSVoice(name=%r, ref_audio_path=%r)" % (
            self.name,
            self.ref_audio_path,
        )


class NeuTTSEngine(BaseEngine):
    """
    NeuTTS engine for on-device text-to-speech with voice cloning.

    The torch Nano backend returns a full waveform. This wrapper pre-encodes the
    reference voice, queues 16-bit PCM chunks promptly after inference, and keeps
    voice/reference inputs explicit for reproducible benchmarks.
    """

    _GGUF_CACHE_FILENAMES = {
        "neuphonic/neutts-nano-q4-gguf": "neutts-nano-Q4_0.gguf",
        "neuphonic/neutts-nano-q8-gguf": "neutts-nano-Q8_0.gguf",
    }

    _BACKBONE_LANGUAGES = {
        "neuphonic/neutts-nano": "en-us",
        "neuphonic/neutts-nano-q4-gguf": "en-us",
        "neuphonic/neutts-nano-q8-gguf": "en-us",
    }

    def __init__(
        self,
        model: str = "neutts-nano",
        backbone_repo: str = "neuphonic/neutts-nano",
        codec_repo: str = "neuphonic/neucodec",
        device: str = "cpu",
        backbone_device: Optional[str] = None,
        codec_device: Optional[str] = None,
        language: Optional[str] = None,
        voice: Optional[NeuTTSVoice] = None,
        voices_dir: Optional[str] = None,
        default_voice: Optional[str] = None,
        preencode_reference: bool = True,
        streaming: bool = True,
        streaming_frames_per_chunk: Optional[int] = None,
        streaming_lookforward: Optional[int] = None,
        streaming_lookback: Optional[int] = None,
        streaming_overlap_frames: Optional[int] = None,
    ):
        super().__init__()
        self.engine_name = "neutts"
        self.model_name = model
        self.backbone_repo = backbone_repo
        self.resolved_backbone_repo = backbone_repo
        self.codec_repo = codec_repo
        self.device = device
        self.backbone_device = backbone_device or device
        self.codec_device = codec_device or device
        self.language = language
        self.voices_dir = voices_dir
        self.default_voice = default_voice
        self.preencode_reference = preencode_reference
        self.streaming = streaming
        self.streaming_frames_per_chunk = streaming_frames_per_chunk
        self.streaming_lookforward = streaming_lookforward
        self.streaming_lookback = streaming_lookback
        self.streaming_overlap_frames = streaming_overlap_frames
        self.sampling_rate = 24000

        self._tts = None
        self._voices: dict[str, NeuTTSVoice] = {}
        self._current_voice: Optional[NeuTTSVoice] = None
        self._reference_cache = {}

        self._init_model()
        if voice is not None:
            self.set_voice(voice)
        self._load_voices()

    def _is_gguf_backbone(self) -> bool:
        lowered = self.backbone_repo.lower()
        return lowered.endswith("gguf") or lowered.endswith(".gguf")

    def _resolve_backbone_repo(self) -> str:
        """Prefer a cached local GGUF file to avoid repeat HF metadata calls."""

        if os.path.isfile(self.backbone_repo):
            return self.backbone_repo

        filename = self._GGUF_CACHE_FILENAMES.get(self.backbone_repo)
        if not filename:
            return self.backbone_repo

        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return self.backbone_repo

        cached = try_to_load_from_cache(self.backbone_repo, filename)
        if isinstance(cached, str) and os.path.exists(cached):
            logging.info("Using cached NeuTTS GGUF backbone: %s", cached)
            return cached

        return self.backbone_repo

    def _resolved_language(self) -> Optional[str]:
        if self.language:
            return self.language
        return self._BACKBONE_LANGUAGES.get(self.backbone_repo)

    def _configure_streaming(self):
        if self._tts is None:
            return
        if self.streaming_frames_per_chunk is not None:
            self._tts.streaming_frames_per_chunk = int(self.streaming_frames_per_chunk)
        if self.streaming_lookforward is not None:
            self._tts.streaming_lookforward = int(self.streaming_lookforward)
        if self.streaming_lookback is not None:
            self._tts.streaming_lookback = int(self.streaming_lookback)
        if self.streaming_overlap_frames is not None:
            self._tts.streaming_overlap_frames = int(self.streaming_overlap_frames)

        frames_per_chunk = int(getattr(self._tts, "streaming_frames_per_chunk", 25))
        hop_length = int(getattr(self._tts, "hop_length", 480))
        self._tts.streaming_stride_samples = frames_per_chunk * hop_length

    def _init_model(self):
        """Initialize the NeuTTS model."""
        try:
            from neutts import NeuTTS

            self.resolved_backbone_repo = self._resolve_backbone_repo()
            language = self._resolved_language()
            logging.info("Initializing NeuTTS with %s", self.resolved_backbone_repo)
            self._tts = NeuTTS(
                backbone_repo=self.resolved_backbone_repo,
                backbone_device=self.backbone_device,
                codec_repo=self.codec_repo,
                codec_device=self.codec_device,
                language=language,
            )
            self._configure_streaming()
            self.sampling_rate = int(getattr(self._tts, "sample_rate", self.sampling_rate))
            logging.info("NeuTTS model initialized successfully")
        except ImportError as exc:
            raise ImportError(
                "NeuTTS is not installed. Install with:\n"
                "  pip install neutts\n"
                "For GGUF streaming and ONNX codec support:\n"
                "  pip install \"neutts[llama,onnx]\"\n"
                "Optional CUDA torch wheel:\n"
                "  pip install --index-url https://download.pytorch.org/whl/cu128 "
                "torch==2.11.0+cu128 torchaudio==2.11.0+cu128"
            ) from exc
        except Exception:
            logging.exception("Failed to initialize NeuTTS")
            raise

    def _load_voices(self):
        """Load voice references from an optional voices directory."""
        if not self.voices_dir:
            return
        if not os.path.exists(self.voices_dir):
            os.makedirs(self.voices_dir, exist_ok=True)
            logging.info("Created voices directory: %s", self.voices_dir)
            return

        for filename in os.listdir(self.voices_dir):
            if not filename.endswith(".wav"):
                continue

            voice_name = os.path.splitext(filename)[0]
            audio_path = os.path.join(self.voices_dir, filename)
            text_path = os.path.join(self.voices_dir, "%s.txt" % voice_name)

            if not os.path.exists(text_path):
                logging.warning("Voice %s missing transcript file: %s", voice_name, text_path)
                continue

            with open(text_path, "r", encoding="utf-8") as transcript_file:
                ref_text = transcript_file.read().strip()

            try:
                self.add_voice(voice_name, audio_path, ref_text)
            except Exception as exc:
                logging.warning("Failed to load voice %s: %s", voice_name, exc)

        if self._voices and self._current_voice is None:
            if self.default_voice and self.default_voice in self._voices:
                self._current_voice = self._voices[self.default_voice]
            else:
                self._current_voice = list(self._voices.values())[0]
            logging.info("Default voice set to: %s", self._current_voice.name)

    def get_stream_info(self):
        """
        Return the PyAudio stream configuration.

        NeuTTS outputs 24 kHz mono 16-bit PCM audio.
        """
        import pyaudio

        return pyaudio.paInt16, 1, self.sampling_rate

    def _voice_cache_key(self, audio_path: str, transcript: str):
        try:
            stat = os.stat(audio_path)
            mtime = stat.st_mtime
            size = stat.st_size
        except OSError:
            mtime = 0
            size = 0
        return (
            os.path.abspath(audio_path),
            mtime,
            size,
            transcript,
            self.backbone_repo,
            self.codec_repo,
            self.backbone_device,
            self.codec_device,
            self._resolved_language(),
        )

    @staticmethod
    def _normalize_ref_codes(ref_codes):
        try:
            import numpy as np
        except ImportError:
            np = None

        if hasattr(ref_codes, "detach"):
            values = ref_codes.detach().cpu().reshape(-1).tolist()
        elif np is not None and isinstance(ref_codes, np.ndarray):
            values = ref_codes.reshape(-1).tolist()
        else:
            values = list(ref_codes)
        return [int(value) for value in values]

    def _encode_reference(self, audio_path: str, transcript: str):
        key = self._voice_cache_key(audio_path, transcript)
        cached = self._reference_cache.get(key)
        if cached is not None:
            return cached

        ref_codes = self._tts.encode_reference(audio_path)
        ref_codes = self._normalize_ref_codes(ref_codes)
        self._reference_cache[key] = ref_codes
        return ref_codes

    def _ensure_voice_encoded(self, voice: NeuTTSVoice) -> NeuTTSVoice:
        if voice.ref_codes is not None:
            voice.ref_codes = self._normalize_ref_codes(voice.ref_codes)
            return voice
        if self.preencode_reference:
            voice.ref_codes = self._encode_reference(voice.ref_audio_path, voice.ref_text)
        return voice

    def _to_pcm16(self, wav) -> bytes:
        import numpy as np

        audio = np.asarray(wav, dtype=np.float32).squeeze()
        if audio.ndim > 1:
            audio = audio[0]
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16).tobytes()

    def _synthesize_streaming(self, text: str, ref_codes, ref_text: str) -> bool:
        for wav_chunk in self._tts.infer_stream(text, ref_codes, ref_text):
            if self.stop_synthesis_event.is_set():
                logging.debug("NeuTTS streaming synthesis stopped")
                return True
            audio_bytes = self._to_pcm16(wav_chunk)
            if audio_bytes:
                self.queue.put(audio_bytes)
        return True

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        """
        Synthesize text and queue PCM chunks.
        """
        super().synthesize(text, sentence_count)
        del sentence_count

        if self.stop_synthesis_event.is_set():
            return True
        if not self._current_voice:
            logging.error("No voice selected for NeuTTS synthesis")
            return False

        try:
            voice = self._ensure_voice_encoded(self._current_voice)
            ref_codes = voice.ref_codes
            if ref_codes is None:
                ref_codes = self._encode_reference(voice.ref_audio_path, voice.ref_text)

            logging.debug("Synthesizing with NeuTTS: %s...", text[:50])
            if self.streaming and self._is_gguf_backbone():
                return self._synthesize_streaming(text, ref_codes, voice.ref_text)

            wav = self._tts.infer(text, ref_codes, voice.ref_text)

            if self.stop_synthesis_event.is_set():
                return True

            audio_bytes = self._to_pcm16(wav)
            chunk_size = 4096
            for offset in range(0, len(audio_bytes), chunk_size):
                if self.stop_synthesis_event.is_set():
                    logging.debug("NeuTTS synthesis stopped before queueing all chunks")
                    return True
                self.queue.put(audio_bytes[offset : offset + chunk_size])

            return True
        except Exception:
            logging.exception("NeuTTS synthesis failed")
            return False

    def get_voices(self) -> list:
        """Get available NeuTTS voices."""
        return list(self._voices.values())

    def set_voice(self, voice: Union[str, NeuTTSVoice]):
        """Set the active reference voice."""
        if isinstance(voice, NeuTTSVoice):
            encoded_voice = self._ensure_voice_encoded(voice)
            self._voices[encoded_voice.name] = encoded_voice
            self._current_voice = encoded_voice
            logging.info("Voice set to: %s", encoded_voice.name)
            return

        if voice in self._voices:
            self._current_voice = self._ensure_voice_encoded(self._voices[voice])
            logging.info("Voice set to: %s", voice)
            return

        for name, candidate in self._voices.items():
            if voice.lower() in name.lower():
                self._current_voice = self._ensure_voice_encoded(candidate)
                logging.info("Voice set to: %s", name)
                return

        logging.warning("Voice '%s' not found", voice)

    def set_voice_parameters(self, **voice_parameters):
        """
        Set voice parameters required by BaseEngine.

        NeuTTS does not currently expose runtime sampling parameters through the
        public package API.
        """
        del voice_parameters

    def add_voice(self, name: str, audio_path: str, transcript: str) -> NeuTTSVoice:
        """Add and optionally pre-encode a reference voice."""
        ref_codes = (
            self._encode_reference(audio_path, transcript)
            if self.preencode_reference
            else None
        )
        voice = NeuTTSVoice(
            name=name,
            ref_audio_path=audio_path,
            ref_text=transcript,
            ref_codes=ref_codes,
        )
        self._voices[name] = voice

        if self._current_voice is None:
            self._current_voice = voice

        logging.info("Added voice: %s", name)
        return voice

    def clone_voice(self, audio_path: str, transcript: str) -> NeuTTSVoice:
        """Create a temporary cloned voice without setting it active."""
        ref_codes = self._encode_reference(audio_path, transcript)
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
        self._reference_cache.clear()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
