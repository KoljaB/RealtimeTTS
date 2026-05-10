import logging
import os
from typing import Optional, Union

import numpy as np

from .base_engine import BaseEngine


class ChatterboxVoice:
    """Reference audio configuration for Chatterbox Turbo voice cloning."""

    def __init__(
        self,
        audio_prompt_path: Optional[str] = None,
        name: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: str = "",
        language: str = "en",
        prompt_wav_path: Optional[str] = None,
    ):
        del ref_text, language
        prompt_path = audio_prompt_path or prompt_wav_path or ref_audio
        if prompt_path and not os.path.exists(prompt_path):
            raise FileNotFoundError(
                "Chatterbox prompt WAV file not found at: %s" % prompt_path
            )
        self.audio_prompt_path = prompt_path
        self.name = name or (
            os.path.splitext(os.path.basename(prompt_path))[0]
            if prompt_path
            else "default"
        )

    def __repr__(self):
        return "ChatterboxVoice(name=%r, audio_prompt_path=%r)" % (
            self.name,
            self.audio_prompt_path,
        )


class ChatterboxEngine(BaseEngine):
    """
    Chatterbox Turbo engine for RealtimeTTS.

    Chatterbox Turbo currently exposes a full-waveform `generate()` API. Stop
    requests are honored before queueing generated audio, so aborting a Zaphod
    utterance discards late chunks instead of releasing stale speech.
    """

    def __init__(
        self,
        voice: Optional[ChatterboxVoice] = None,
        device: str = "cuda",
        model_path: Optional[str] = None,
        repetition_penalty: float = 1.2,
        min_p: float = 0.0,
        top_p: float = 0.95,
        exaggeration: float = 0.0,
        cfg_weight: float = 0.0,
        temperature: float = 0.8,
        top_k: int = 1000,
        norm_loudness: bool = True,
        trim_silence: bool = True,
        silence_threshold: float = 0.005,
        extra_start_ms: int = 10,
        extra_end_ms: int = 15,
        fade_in_ms: int = 10,
        fade_out_ms: int = 10,
    ):
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
        except ImportError as exc:
            raise ImportError(
                "Failed to import Chatterbox Turbo. Install it with: "
                "pip install chatterbox-tts. Original error: %s" % exc
            ) from exc

        self.device = device
        self.model_path = model_path
        self.repetition_penalty = repetition_penalty
        self.min_p = min_p
        self.top_p = top_p
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.top_k = top_k
        self.norm_loudness = norm_loudness
        self.trim_silence = trim_silence
        self.silence_threshold = silence_threshold
        self.extra_start_ms = extra_start_ms
        self.extra_end_ms = extra_end_ms
        self.fade_in_ms = fade_in_ms
        self.fade_out_ms = fade_out_ms
        self.voice = None
        self._prepared_voice_path = None

        logging.info("Loading Chatterbox Turbo model on %s", device)
        if model_path:
            self._model = ChatterboxTurboTTS.from_local(model_path, device=device)
        else:
            self._model = ChatterboxTurboTTS.from_pretrained(device=device)
        self.sampling_rate = int(getattr(self._model, "sr", 24000))

        if voice is not None:
            self.set_voice(voice)
        self.post_init()

    def post_init(self):
        self.engine_name = "chatterbox"

    def get_stream_info(self):
        import pyaudio

        return pyaudio.paInt16, 1, self.sampling_rate

    def get_voices(self):
        return []

    def set_voice(self, voice: Union[ChatterboxVoice, str]):
        if isinstance(voice, str):
            voice = ChatterboxVoice(audio_prompt_path=voice)
        if not isinstance(voice, ChatterboxVoice):
            raise TypeError("voice must be a ChatterboxVoice instance or path string.")

        if not voice.audio_prompt_path:
            self.voice = voice
            return

        prompt_path = os.path.abspath(voice.audio_prompt_path)
        if self._prepared_voice_path != prompt_path:
            logging.info("Encoding Chatterbox prompt: %s", voice.audio_prompt_path)
            self._model.prepare_conditionals(
                voice.audio_prompt_path,
                exaggeration=self.exaggeration,
                norm_loudness=self.norm_loudness,
            )
            self._prepared_voice_path = prompt_path
        self.voice = voice

    def set_voice_parameters(self, **voice_parameters):
        valid_params = {
            "repetition_penalty",
            "min_p",
            "top_p",
            "exaggeration",
            "cfg_weight",
            "temperature",
            "top_k",
            "norm_loudness",
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
        if "exaggeration" in voice_parameters and self.voice is not None:
            self._prepared_voice_path = None
            self.set_voice(self.voice)

    def _to_float_mono(self, wav):
        if hasattr(wav, "detach"):
            audio = wav.detach().float().cpu().numpy()
        else:
            audio = np.asarray(wav, dtype=np.float32)
        audio = np.squeeze(audio)
        if audio.ndim > 1:
            audio = audio[0]
        return np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)
        del sentence_count

        if self.stop_synthesis_event.is_set():
            return True

        try:
            if self.voice is not None and self.voice.audio_prompt_path:
                self.set_voice(self.voice)

            wav = self._model.generate(
                text,
                repetition_penalty=self.repetition_penalty,
                min_p=self.min_p,
                top_p=self.top_p,
                audio_prompt_path=None,
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight,
                temperature=self.temperature,
                top_k=self.top_k,
                norm_loudness=self.norm_loudness,
            )

            if self.stop_synthesis_event.is_set():
                return True

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
            self.queue.put((audio * 32767).astype(np.int16).tobytes())
            return True
        except Exception:
            logging.exception("Chatterbox synthesis failed")
            return False

    def shutdown(self):
        self._prepared_voice_path = None
        if hasattr(self, "_model"):
            del self._model
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("ChatterboxEngine shutdown.")
