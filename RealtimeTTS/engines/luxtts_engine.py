import logging
import os
import sys
from contextlib import nullcontext
from typing import Optional, Union

import numpy as np
import torch

from .base_engine import BaseEngine


def _disable_torchcodec_for_windows():
    """
    Avoid Transformers selecting torchcodec on Windows.

    The LuxTTS Gradio Windows setup disables this to avoid DLL/ffmpeg mismatch
    crashes during ASR prompt handling.
    """

    try:
        from transformers.utils import import_utils
    except Exception:
        return
    if hasattr(import_utils, "_torchcodec_available"):
        import_utils._torchcodec_available = False


class LuxTTSVoice:
    """
    Voice prompt configuration for LuxTTS zero-shot voice cloning.
    """

    def __init__(
        self,
        prompt_wav_path: str,
        prompt_text: str = "",
        name: Optional[str] = None,
    ):
        if not os.path.exists(prompt_wav_path):
            raise FileNotFoundError(f"Prompt WAV file not found at: {prompt_wav_path}")
        self.prompt_wav_path = prompt_wav_path
        self.prompt_text = prompt_text
        self.name = name or os.path.splitext(os.path.basename(prompt_wav_path))[0]

    def __repr__(self):
        return (
            "LuxTTSVoice(name=%r, prompt_wav=%r)"
            % (self.name, self.prompt_wav_path)
        )


class LuxTTSEngine(BaseEngine):
    """
    LuxTTS engine for RealtimeTTS.

    LuxTTS is based on ZipVoice/flow matching and returns a full utterance.
    It does not currently expose native mid-inference cancellation; stop requests
    are honored before queueing audio if they arrive while generation is running.
    """

    def __init__(
        self,
        lux_root: Optional[str] = None,
        voice: Optional[LuxTTSVoice] = None,
        model_path: str = "YatharthS/LuxTTS",
        device: str = "cuda",
        threads: int = 4,
        num_steps: int = 4,
        guidance_scale: float = 3.0,
        t_shift: float = 0.5,
        speed: float = 1.0,
        return_smooth: bool = False,
        prompt_duration: float = 5.0,
        target_rms: float = 0.01,
        use_autocast: bool = False,
        autocast_dtype: str = "float16",
    ):
        self.lux_root = None
        self._added_sys_path = None
        if lux_root:
            self.lux_root = os.path.abspath(lux_root)
            if not os.path.isdir(self.lux_root):
                raise NotADirectoryError(
                    "lux_root is not a valid directory: %s" % self.lux_root
                )
            if self.lux_root not in sys.path:
                sys.path.insert(0, self.lux_root)
                self._added_sys_path = self.lux_root

        try:
            from zipvoice.luxvoice import LuxTTS
        except ImportError as exc:
            raise ImportError(
                "Failed to import LuxTTS. Install the LuxTTS source package or "
                "pass lux_root pointing to https://github.com/ysharma3501/LuxTTS. "
                "Original error: %s" % exc
            ) from exc

        if voice is None:
            raise ValueError("LuxTTSEngine requires a LuxTTSVoice prompt.")

        self.model_path = model_path
        self.device = device
        self.threads = threads
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.t_shift = t_shift
        self.speed = speed
        self.return_smooth = return_smooth
        self.prompt_duration = prompt_duration
        self.target_rms = target_rms
        self.use_autocast = use_autocast
        self.autocast_dtype = autocast_dtype
        self.sampling_rate = 24000 if self.return_smooth else 48000

        self.voice = None
        self._encoded_prompt = None
        self._prompt_cache = {}

        _disable_torchcodec_for_windows()
        logging.info("Loading LuxTTS model from %s on %s", model_path, device)
        self._model = LuxTTS(model_path, device=device, threads=threads)
        self.set_voice(voice)
        self.post_init()

    def post_init(self):
        self.engine_name = "luxtts"

    def get_stream_info(self):
        import pyaudio

        return pyaudio.paInt16, 1, self.sampling_rate

    def _prompt_cache_key(self, voice: LuxTTSVoice):
        try:
            mtime = os.path.getmtime(voice.prompt_wav_path)
        except OSError:
            mtime = 0
        return (
            os.path.abspath(voice.prompt_wav_path),
            mtime,
            self.prompt_duration,
            self.target_rms,
        )

    def _encode_prompt(self, voice: LuxTTSVoice):
        key = self._prompt_cache_key(voice)
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached

        logging.info("Encoding LuxTTS prompt: %s", voice.prompt_wav_path)
        encoded = self._model.encode_prompt(
            voice.prompt_wav_path,
            duration=self.prompt_duration,
            rms=self.target_rms,
        )
        self._prompt_cache[key] = encoded
        return encoded

    def set_voice(self, voice: Union[LuxTTSVoice, str]):
        if not isinstance(voice, LuxTTSVoice):
            raise TypeError("voice must be a LuxTTSVoice instance.")

        if (
            self.voice is not None
            and os.path.abspath(self.voice.prompt_wav_path)
            == os.path.abspath(voice.prompt_wav_path)
            and self._encoded_prompt is not None
        ):
            self.voice = voice
            return

        self.voice = voice
        self._encoded_prompt = self._encode_prompt(voice)

    def get_voices(self):
        return []

    def set_voice_parameters(self, **voice_parameters):
        valid_params = {
            "num_steps",
            "guidance_scale",
            "t_shift",
            "speed",
            "return_smooth",
            "prompt_duration",
            "target_rms",
            "use_autocast",
            "autocast_dtype",
        }
        for param, value in voice_parameters.items():
            if param not in valid_params:
                continue
            setattr(self, param, value)
            if param == "return_smooth":
                self.sampling_rate = 24000 if self.return_smooth else 48000
            if param in {"prompt_duration", "target_rms"}:
                self._encoded_prompt = self._encode_prompt(self.voice)

    @torch.inference_mode()
    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)
        del sentence_count

        if self.stop_synthesis_event.is_set():
            return True
        if self._encoded_prompt is None:
            logging.error("LuxTTS voice prompt is not encoded.")
            return False

        try:
            autocast_context = nullcontext()
            if self.use_autocast and self.device.startswith("cuda"):
                dtype = torch.float16
                if self.autocast_dtype == "bfloat16":
                    dtype = torch.bfloat16
                autocast_context = torch.amp.autocast("cuda", dtype=dtype)

            with autocast_context:
                wav = self._model.generate_speech(
                    text,
                    self._encoded_prompt,
                    num_steps=self.num_steps,
                    guidance_scale=self.guidance_scale,
                    t_shift=self.t_shift,
                    speed=self.speed,
                    return_smooth=self.return_smooth,
                )

            if self.stop_synthesis_event.is_set():
                return True

            if hasattr(wav, "detach"):
                audio = wav.detach().float().cpu().numpy()
            else:
                audio = np.asarray(wav, dtype=np.float32)
            audio = np.squeeze(audio)
            if audio.ndim > 1:
                audio = audio[0]
            audio = np.clip(audio, -1.0, 1.0)
            self.queue.put((audio * 32767).astype(np.int16).tobytes())
            return True
        except Exception:
            logging.exception("LuxTTS synthesis failed")
            return False

    def shutdown(self):
        self._prompt_cache.clear()
        self._encoded_prompt = None
        if hasattr(self, "_model"):
            del self._model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._added_sys_path and self._added_sys_path in sys.path:
            sys.path.remove(self._added_sys_path)
        logging.info("LuxTTSEngine shutdown.")
