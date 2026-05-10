import logging
from typing import Optional

import numpy as np

from .base_engine import BaseEngine


class SopranoVoice:
    """Placeholder voice object for Soprano's current single-voice model."""

    def __init__(self, name: str = "default"):
        self.name = name

    def __repr__(self):
        return "SopranoVoice(name=%r)" % self.name


class SopranoEngine(BaseEngine):
    """
    Soprano TTS engine for RealtimeTTS.

    Soprano is currently a single-voice English TTS model. The upstream package
    exposes both streaming and full-waveform APIs; this wrapper queues float
    output as 16-bit PCM chunks for RealtimeTTS.
    """

    def __init__(
        self,
        voice: Optional[SopranoVoice] = None,
        backend: str = "auto",
        device: str = "auto",
        model_path: Optional[str] = None,
        cache_size_mb: int = 100,
        decoder_batch_size: int = 1,
        chunk_size: int = 1,
        top_p: float = 0.95,
        temperature: float = 0.0,
        repetition_penalty: float = 1.2,
        streaming: bool = True,
        trim_silence: bool = False,
        silence_threshold: float = 0.005,
        extra_start_ms: int = 0,
        extra_end_ms: int = 10,
        fade_in_ms: int = 5,
        fade_out_ms: int = 10,
    ):
        try:
            from soprano import SopranoTTS
        except ImportError as exc:
            raise ImportError(
                "Failed to import SopranoTTS. Install it with: "
                "pip install soprano-tts. Original error: %s" % exc
            ) from exc

        self.voice = voice or SopranoVoice()
        self.backend = backend
        self.device = device
        self.model_path = model_path
        self.cache_size_mb = cache_size_mb
        self.decoder_batch_size = decoder_batch_size
        self.chunk_size = chunk_size
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.streaming = streaming
        self.trim_silence = trim_silence
        self.silence_threshold = silence_threshold
        self.extra_start_ms = extra_start_ms
        self.extra_end_ms = extra_end_ms
        self.fade_in_ms = fade_in_ms
        self.fade_out_ms = fade_out_ms
        self.sampling_rate = 32000

        logging.info("Loading SopranoTTS model on %s with %s backend", device, backend)
        self._model = SopranoTTS(
            backend=backend,
            device=device,
            cache_size_mb=cache_size_mb,
            decoder_batch_size=decoder_batch_size,
            model_path=model_path,
        )
        self.post_init()

    def post_init(self):
        self.engine_name = "soprano"

    def get_stream_info(self):
        import pyaudio

        return pyaudio.paInt16, 1, self.sampling_rate

    def get_voices(self):
        return [SopranoVoice()]

    def set_voice(self, voice):
        if isinstance(voice, str):
            voice = SopranoVoice(voice)
        if not isinstance(voice, SopranoVoice):
            raise TypeError("voice must be a SopranoVoice instance or name string.")
        self.voice = voice

    def set_voice_parameters(self, **voice_parameters):
        valid_params = {
            "backend",
            "device",
            "cache_size_mb",
            "decoder_batch_size",
            "chunk_size",
            "top_p",
            "temperature",
            "repetition_penalty",
            "streaming",
            "trim_silence",
            "silence_threshold",
            "extra_start_ms",
            "extra_end_ms",
            "fade_in_ms",
            "fade_out_ms",
        }
        for param, value in voice_parameters.items():
            if param in valid_params:
                setattr(self, param, value)

    def _to_float_mono(self, wav):
        if hasattr(wav, "detach"):
            audio = wav.detach().float().cpu().numpy()
        else:
            audio = np.asarray(wav, dtype=np.float32)
        audio = np.squeeze(audio)
        if audio.ndim > 1:
            audio = audio[0]
        return np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)

    def _queue_audio(self, wav) -> None:
        audio = self._to_float_mono(wav)
        if self.trim_silence:
            audio = self._trim_silence(
                audio,
                sample_rate=self.sampling_rate,
                silence_threshold=self.silence_threshold,
                extra_start_ms=self.extra_start_ms,
                extra_end_ms=self.extra_end_ms,
                fade_in_ms=self.fade_in_ms,
                fade_out_ms=self.fade_out_ms,
            )
        if audio.size:
            self.queue.put((audio * 32767).astype(np.int16).tobytes())

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)
        del sentence_count

        if self.stop_synthesis_event.is_set():
            return True

        try:
            if not self.streaming:
                wav = self._model.infer(
                    text,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                )
                if not self.stop_synthesis_event.is_set():
                    self._queue_audio(wav)
                return True

            for wav_chunk in self._model.infer_stream(
                text,
                chunk_size=self.chunk_size,
                top_p=self.top_p,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
            ):
                if self.stop_synthesis_event.is_set():
                    return True
                self._queue_audio(wav_chunk)
            return True
        except Exception:
            logging.exception("SopranoTTS synthesis failed")
            return False

    def shutdown(self):
        if hasattr(self, "_model"):
            del self._model
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("SopranoEngine shutdown.")
