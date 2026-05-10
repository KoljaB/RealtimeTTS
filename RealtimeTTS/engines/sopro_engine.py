import logging
import os
from typing import Optional, Union

import numpy as np

from .base_engine import BaseEngine


class SoproTTSVoice:
    """Reference audio configuration for SoproTTS zero-shot voice cloning."""

    def __init__(
        self,
        ref_audio_path: Optional[str] = None,
        name: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: str = "",
        language: str = "en",
        prompt_wav_path: Optional[str] = None,
    ):
        del ref_text, language
        prompt_path = ref_audio_path or ref_audio or prompt_wav_path
        if prompt_path and not os.path.exists(prompt_path):
            raise FileNotFoundError("SoproTTS reference WAV file not found at: %s" % prompt_path)
        self.ref_audio_path = prompt_path
        self.name = name or (
            os.path.splitext(os.path.basename(prompt_path))[0]
            if prompt_path
            else "default"
        )

    def __repr__(self):
        return "SoproTTSVoice(name=%r, ref_audio_path=%r)" % (
            self.name,
            self.ref_audio_path,
        )


class SoproTTSEngine(BaseEngine):
    """
    SoproTTS engine for RealtimeTTS.

    Sopro exposes a native streaming API. This wrapper prepares the reference
    voice once, then queues each generated audio chunk as 16-bit PCM.
    """

    def __init__(
        self,
        voice: Optional[SoproTTSVoice] = None,
        model_name: str = "samuel-vitorino/sopro",
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        device: str = "cuda",
        max_frames: int = 400,
        top_p: float = 0.9,
        temperature: float = 1.05,
        anti_loop: bool = True,
        style_strength: Optional[float] = None,
        ref_seconds: Optional[float] = 12.0,
        chunk_frames: int = 6,
        nar_context_frames: Optional[int] = None,
        min_gen_frames: Optional[int] = None,
        streaming: bool = True,
        lowpass_cutoff_hz: Optional[float] = None,
        trim_silence: bool = False,
        silence_threshold: float = 0.005,
        extra_start_ms: int = 0,
        extra_end_ms: int = 10,
        fade_in_ms: int = 5,
        fade_out_ms: int = 10,
    ):
        try:
            from sopro import SoproTTS
            from sopro.constants import TARGET_SR
        except ImportError as exc:
            raise ImportError(
                "Failed to import SoproTTS. Install it with: pip install sopro. "
                "Original error: %s" % exc
            ) from exc

        self.model_name = model_name
        self.revision = revision
        self.cache_dir = cache_dir
        self.token = token
        self.device = device
        self.max_frames = max_frames
        self.top_p = top_p
        self.temperature = temperature
        self.anti_loop = anti_loop
        self.style_strength = style_strength
        self.ref_seconds = ref_seconds
        self.chunk_frames = chunk_frames
        self.nar_context_frames = nar_context_frames
        self.min_gen_frames = min_gen_frames
        self.streaming = streaming
        self.lowpass_cutoff_hz = lowpass_cutoff_hz
        self.trim_silence = trim_silence
        self.silence_threshold = silence_threshold
        self.extra_start_ms = extra_start_ms
        self.extra_end_ms = extra_end_ms
        self.fade_in_ms = fade_in_ms
        self.fade_out_ms = fade_out_ms
        self.sampling_rate = int(TARGET_SR)
        self.voice = None
        self._prepared_ref = None
        self._prepared_voice_key = None
        self._reference_cache = {}

        if cache_dir:
            os.environ["HF_HOME"] = cache_dir
            os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")

        logging.info("Loading SoproTTS model %s on %s", model_name, device)
        self._model = SoproTTS.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            device=device,
        )

        if voice is not None:
            self.set_voice(voice)
        self.post_init()

    def post_init(self):
        self.engine_name = "sopro"

    def get_stream_info(self):
        import pyaudio

        return pyaudio.paInt16, 1, self.sampling_rate

    def get_voices(self):
        return []

    def _voice_cache_key(self, voice: SoproTTSVoice):
        try:
            mtime = os.path.getmtime(voice.ref_audio_path)
        except OSError:
            mtime = 0
        return (
            os.path.abspath(voice.ref_audio_path),
            mtime,
            self.ref_seconds,
        )

    def _prepare_reference(self, voice: SoproTTSVoice):
        if not voice.ref_audio_path:
            raise ValueError("SoproTTS requires a reference audio path.")
        key = self._voice_cache_key(voice)
        cached = self._reference_cache.get(key)
        if cached is not None:
            return cached

        logging.info("Encoding SoproTTS reference: %s", voice.ref_audio_path)
        prepared = self._model.prepare_reference(
            ref_audio_path=voice.ref_audio_path,
            ref_seconds=self.ref_seconds,
        )
        self._reference_cache[key] = prepared
        return prepared

    def set_voice(self, voice: Union[SoproTTSVoice, str]):
        if isinstance(voice, str):
            voice = SoproTTSVoice(ref_audio_path=voice)
        if not isinstance(voice, SoproTTSVoice):
            raise TypeError("voice must be a SoproTTSVoice instance or path string.")

        key = self._voice_cache_key(voice) if voice.ref_audio_path else None
        if self.voice is not None and key == self._prepared_voice_key:
            self.voice = voice
            return

        self.voice = voice
        self._prepared_ref = self._prepare_reference(voice)
        self._prepared_voice_key = key

    def set_voice_parameters(self, **voice_parameters):
        valid_params = {
            "max_frames",
            "top_p",
            "temperature",
            "anti_loop",
            "style_strength",
            "ref_seconds",
            "chunk_frames",
            "nar_context_frames",
            "min_gen_frames",
            "streaming",
            "lowpass_cutoff_hz",
            "trim_silence",
            "silence_threshold",
            "extra_start_ms",
            "extra_end_ms",
            "fade_in_ms",
            "fade_out_ms",
        }
        should_reprepare = False
        for param, value in voice_parameters.items():
            if param not in valid_params:
                continue
            setattr(self, param, value)
            if param == "ref_seconds":
                should_reprepare = True
        if should_reprepare and self.voice is not None:
            self._prepared_voice_key = None
            self._prepared_ref = self._prepare_reference(self.voice)

    def _to_float_mono(self, wav):
        if hasattr(wav, "detach"):
            audio = wav.detach().float().cpu().numpy()
        else:
            audio = np.asarray(wav, dtype=np.float32)
        audio = np.squeeze(audio)
        if audio.ndim > 1:
            audio = audio[0]
        return np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)

    def _lowpass(self, audio):
        cutoff = self.lowpass_cutoff_hz
        if cutoff is None or cutoff <= 0:
            return audio
        nyquist = self.sampling_rate / 2.0
        if cutoff >= nyquist:
            return audio
        if audio.size < 4:
            return audio

        spectrum = np.fft.rfft(audio.astype(np.float32, copy=False))
        freqs = np.fft.rfftfreq(audio.size, d=1.0 / self.sampling_rate)
        spectrum[freqs > cutoff] = 0
        filtered = np.fft.irfft(spectrum, n=audio.size).astype(np.float32, copy=False)
        return np.clip(filtered, -1.0, 1.0)

    def _queue_audio(self, wav) -> None:
        audio = self._to_float_mono(wav)
        audio = self._lowpass(audio)
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
        if self._prepared_ref is None:
            logging.error("SoproTTS reference voice is not prepared.")
            return False

        try:
            if not self.streaming:
                wav = self._model.synthesize(
                    text,
                    ref=self._prepared_ref,
                    max_frames=self.max_frames,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    anti_loop=self.anti_loop,
                    style_strength=self.style_strength,
                    min_gen_frames=self.min_gen_frames,
                )
                if not self.stop_synthesis_event.is_set():
                    self._queue_audio(wav)
                return True

            for wav_chunk in self._model.stream(
                text,
                ref=self._prepared_ref,
                max_frames=self.max_frames,
                top_p=self.top_p,
                temperature=self.temperature,
                anti_loop=self.anti_loop,
                style_strength=self.style_strength,
                chunk_frames=self.chunk_frames,
                nar_context_frames=self.nar_context_frames,
                min_gen_frames=self.min_gen_frames,
            ):
                if self.stop_synthesis_event.is_set():
                    return True

                self._queue_audio(wav_chunk)
            return True
        except Exception:
            logging.exception("SoproTTS synthesis failed")
            return False

    def shutdown(self):
        self._reference_cache.clear()
        self._prepared_ref = None
        if hasattr(self, "_model"):
            del self._model
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("SoproTTSEngine shutdown.")
