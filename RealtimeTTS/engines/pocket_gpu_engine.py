"""
PocketTTS GPU engine for RealtimeTTS.

This engine targets the CUDA-capable ``pocket-tts-gpu`` fork rather than the
official CPU-first ``pocket-tts`` package. Keep it separate from
``PocketTTSEngine`` so CPU installs and release expectations stay predictable.
"""

from __future__ import annotations

import hashlib
import os
import time
import traceback
from pathlib import Path
from queue import Queue
from typing import Any, Optional, Union

import numpy as np
import pyaudio

from .base_engine import BaseEngine


class PocketTTSGpuVoice:
    """Voice descriptor for the PocketTTS GPU engine."""

    BUILTIN_VOICES = [
        "alba",
        "marius",
        "javert",
        "jean",
        "fantine",
        "cosette",
        "eponine",
        "azelma",
    ]

    def __init__(
        self,
        name: str,
        audio_prompt_path: Optional[str] = None,
        state_path: Optional[str] = None,
    ) -> None:
        self.name = name
        self.audio_prompt_path = audio_prompt_path
        self.state_path = state_path
        self.is_builtin = name in self.BUILTIN_VOICES and audio_prompt_path is None

    def __repr__(self) -> str:
        if self.state_path:
            return f"PocketTTSGpuVoice(name='{self.name}', state='{self.state_path}')"
        if self.audio_prompt_path:
            return f"PocketTTSGpuVoice(name='{self.name}', cloned_from='{self.audio_prompt_path}')"
        return f"PocketTTSGpuVoice(name='{self.name}', builtin={self.is_builtin})"


class PocketTTSGpuEngine(BaseEngine):
    """RealtimeTTS engine using the CUDA PocketTTS fork."""

    def __init__(
        self,
        voice: Union[str, PocketTTSGpuVoice] = "alba",
        device: str = "cuda",
        variant: str = "b6369a24",
        model_config: Optional[str] = None,
        voice_cache_dir: Optional[str] = None,
        cache_voice_states: bool = True,
        teacher_forcing: bool = False,
        frames_after_eos: Optional[int] = None,
        trim_silence: bool = False,
        silence_threshold: float = 0.005,
        extra_start_ms: int = 0,
        extra_end_ms: int = 10,
        fade_in_ms: int = 5,
        fade_out_ms: int = 10,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.engine_name = "pocket_tts_gpu"
        self.queue = Queue()
        self.debug = debug
        self.device = device
        self.variant = variant
        self.model_config = Path(model_config) if model_config else None
        self.voice_cache_dir = self._resolve_cache_dir(voice_cache_dir)
        self.cache_voice_states = cache_voice_states
        self.teacher_forcing = teacher_forcing
        self.frames_after_eos = frames_after_eos
        self.trim_silence = trim_silence
        self.silence_threshold = silence_threshold
        self.extra_start_ms = extra_start_ms
        self.extra_end_ms = extra_end_ms
        self.fade_in_ms = fade_in_ms
        self.fade_out_ms = fade_out_ms

        self.model = None
        self.sample_rate = 24000
        self.current_voice: Optional[PocketTTSGpuVoice] = None
        self.current_voice_state: Optional[dict[str, Any]] = None
        self._voice_states: dict[str, dict[str, Any]] = {}

        self._load_model()
        self.set_voice(voice)

    def post_init(self) -> None:
        self.engine_name = "pocket_tts_gpu"

    def _resolve_cache_dir(self, voice_cache_dir: Optional[str]) -> Optional[Path]:
        if not voice_cache_dir or voice_cache_dir.strip().lower() in {"none", "off", "false"}:
            return None
        requested = Path(voice_cache_dir)
        requested.mkdir(parents=True, exist_ok=True)
        return requested

    def _load_model(self) -> None:
        try:
            import torch
            from pocket_tts import TTSModel
            from pocket_tts.default_parameters import (
                DEFAULT_EOS_THRESHOLD,
                DEFAULT_LSD_DECODE_STEPS,
                DEFAULT_NOISE_CLAMP,
                DEFAULT_TEMPERATURE,
            )
            from pocket_tts.utils.config import load_config
        except ImportError as exc:
            raise ImportError(
                "PocketTTS GPU dependencies are missing. Install the CUDA fork first: "
                "pip install git+https://github.com/Deveraux-Parker/kutai100temp.git@6beddc19c480da9ced9733ba0bb2f199f6e22ab4#subdirectory=pocket-tts-gpu"
            ) from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "PocketTTS GPU requested CUDA, but torch.cuda.is_available() is false. "
                "Install a CUDA-enabled PyTorch build for your driver/CUDA runtime "
                "and verify it with: python -c \"import torch; print(torch.cuda.is_available())\""
            )

        config_path = self.model_config or self._auto_local_config()
        if config_path:
            config = load_config(config_path)
            self.model = TTSModel._from_pydantic_config_with_weights(
                config,
                DEFAULT_TEMPERATURE,
                DEFAULT_LSD_DECODE_STEPS,
                DEFAULT_NOISE_CLAMP,
                DEFAULT_EOS_THRESHOLD,
            )
        else:
            self.model = TTSModel.load_model(self.variant)

        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = int(self.model.sample_rate)
        if self.debug:
            print(
                f"[PocketTTSGpuEngine] Model loaded on {self.model.device}; "
                f"sample_rate={self.sample_rate}"
            )

    def _auto_local_config(self) -> Optional[Path]:
        if self.voice_cache_dir is None:
            return None

        snapshot = self._find_kyutai_snapshot()
        if snapshot is None:
            return None

        weights_path = snapshot / "tts_b6369a24.safetensors"
        tokenizer_path = snapshot / "languages" / "english" / "tokenizer.model"
        if not weights_path.exists() or not tokenizer_path.exists():
            return None

        source_config = self._package_config_path()
        if not source_config.exists():
            return None

        text = source_config.read_text(encoding="utf-8-sig")
        text = text.replace(
            "hf://kyutai/pocket-tts/tts_b6369a24.safetensors",
            str(weights_path).replace(os.sep, "/"),
        )
        text = text.replace(
            "hf://kyutai/pocket-tts-without-voice-cloning/tokenizer.model",
            str(tokenizer_path).replace(os.sep, "/"),
        )
        config_path = self.voice_cache_dir / "configs" / f"{self.variant}_local.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(text, encoding="utf-8")
        return config_path

    def _package_config_path(self) -> Path:
        import pocket_tts

        return Path(pocket_tts.__file__).resolve().parent / "config" / f"{self.variant}.yaml"

    def _find_kyutai_snapshot(self) -> Optional[Path]:
        hf_home = Path(os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface")
        snapshots_dir = hf_home / "hub" / "models--kyutai--pocket-tts" / "snapshots"
        if not snapshots_dir.exists():
            return None
        candidates = sorted(
            snapshots_dir.glob("*/tts_b6369a24.safetensors"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        return candidates[0].parent if candidates else None

    def _voice_cache_key(
        self,
        voice_name: str,
        audio_path: Optional[str],
        state_path: Optional[str],
    ) -> str:
        return f"{voice_name}:{state_path or audio_path or 'builtin'}"

    def _cached_state_path(self, voice_name: str, audio_path: str) -> Optional[Path]:
        if not self.cache_voice_states or self.voice_cache_dir is None:
            return None
        source = Path(audio_path)
        try:
            stat = source.stat()
            cache_id = f"{source.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{self.variant}"
        except OSError:
            cache_id = f"{source}:{self.variant}"
        digest = hashlib.sha256(cache_id.encode("utf-8")).hexdigest()[:16]
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in voice_name)
        return self.voice_cache_dir / f"{safe_name}_{digest}.safetensors"

    def _pcm16_prompt_path(self, voice_name: str, audio_path: str) -> str:
        if self.voice_cache_dir is None:
            return audio_path
        source = Path(audio_path)
        try:
            stat = source.stat()
            cache_id = f"{source.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            cache_id = str(source)
        digest = hashlib.sha256(cache_id.encode("utf-8")).hexdigest()[:16]
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in voice_name)
        pcm_path = self.voice_cache_dir / "prompt_wavs" / f"{safe_name}_{digest}.wav"
        if pcm_path.exists():
            return str(pcm_path)
        pcm_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from scipy.io import wavfile

            sample_rate, data = wavfile.read(str(source))
            audio = np.asarray(data)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if np.issubdtype(audio.dtype, np.floating):
                audio_float = np.clip(audio.astype(np.float32), -1.0, 1.0)
            else:
                info = np.iinfo(audio.dtype)
                scale = float(max(abs(info.min), abs(info.max)))
                audio_float = audio.astype(np.float32) / scale
            audio_int16 = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
            wavfile.write(str(pcm_path), sample_rate, audio_int16)
            return str(pcm_path)
        except Exception:
            if self.debug:
                print(f"[PocketTTSGpuEngine] Could not convert prompt WAV to PCM16: {audio_path}")
            return audio_path

    def _save_voice_state(self, voice_state: dict[str, Any], state_path: Path) -> None:
        try:
            import torch
            from safetensors.torch import save_file

            tensors = {}
            for outer_key, inner in voice_state.items():
                if isinstance(inner, dict):
                    for inner_key, value in inner.items():
                        if isinstance(value, torch.Tensor):
                            tensors[f"{outer_key}.{inner_key}"] = value.detach().cpu()
                elif isinstance(inner, torch.Tensor):
                    tensors[outer_key] = inner.detach().cpu()
            if tensors:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(tensors, str(state_path))
                if self.debug:
                    print(f"[PocketTTSGpuEngine] Cached voice state: {state_path}")
        except Exception:
            if self.debug:
                print(f"[PocketTTSGpuEngine] Could not cache voice state: {state_path}")

    def _load_voice_state(self, state_path: Path) -> dict[str, Any]:
        import torch
        from safetensors.torch import load_file

        flat = load_file(str(state_path), device=self.device)
        state: dict[str, Any] = {}
        for key, value in flat.items():
            if "." in key:
                outer, inner = key.split(".", 1)
                state.setdefault(outer, {})[inner] = value.to(self.device)
            else:
                state[key] = value.to(self.device)
        for value in state.values():
            if isinstance(value, dict):
                for inner_key, tensor in value.items():
                    if isinstance(tensor, torch.Tensor):
                        value[inner_key] = tensor.to(self.device)
        return state

    def _get_voice_state(self, voice: Union[str, PocketTTSGpuVoice]) -> dict[str, Any]:
        if isinstance(voice, PocketTTSGpuVoice):
            voice_name = voice.name
            audio_path = voice.audio_prompt_path
            state_path = voice.state_path
        else:
            voice_name = voice
            audio_path = None
            state_path = None

        cache_key = self._voice_cache_key(voice_name, audio_path, state_path)
        if cache_key in self._voice_states:
            return self._voice_states[cache_key]

        if state_path:
            voice_state = self._load_voice_state(Path(state_path))
        elif audio_path:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Voice audio file not found: {audio_path}")
            cached_state_path = self._cached_state_path(voice_name, audio_path)
            if cached_state_path is not None and cached_state_path.exists():
                voice_state = self._load_voice_state(cached_state_path)
            else:
                prompt_path = self._pcm16_prompt_path(voice_name, audio_path)
                voice_state = self.model.get_state_for_audio_prompt(prompt_path)
                if cached_state_path is not None:
                    self._save_voice_state(voice_state, cached_state_path)
        else:
            if voice_name not in PocketTTSGpuVoice.BUILTIN_VOICES:
                raise ValueError(
                    f"Unknown voice: {voice_name}. Available voices: {PocketTTSGpuVoice.BUILTIN_VOICES}"
                )
            voice_state = self.model.get_state_for_audio_prompt(voice_name)

        self._voice_states[cache_key] = voice_state
        return voice_state

    def get_stream_info(self):
        return (pyaudio.paInt16, 1, self.sample_rate)

    def _to_numpy_audio(self, audio: Any) -> np.ndarray:
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu()
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        audio_float32 = np.asarray(audio, dtype=np.float32)
        if audio_float32.ndim > 1:
            audio_float32 = audio_float32.squeeze()
        return audio_float32.reshape(-1)

    def _queue_audio(self, audio_float32: np.ndarray) -> int:
        if audio_float32.size == 0:
            return 0
        audio_float32 = np.clip(audio_float32, -1.0, 1.0)
        audio_int16 = (audio_float32 * 32767).astype(np.int16).tobytes()
        self.audio_duration += len(audio_float32) / self.sample_rate
        self.queue.put(audio_int16)
        return len(audio_int16)

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)
        if self.stop_synthesis_event.is_set():
            return True
        if not text or not text.strip():
            return True

        start_time = time.perf_counter()
        try:
            for audio_chunk in self.model.generate_audio_stream(
                self.current_voice_state,
                text,
                frames_after_eos=self.frames_after_eos,
                teacher_forcing=self.teacher_forcing,
            ):
                if self.stop_synthesis_event.is_set():
                    return True
                audio_float32 = self._to_numpy_audio(audio_chunk)
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
                self._queue_audio(audio_float32)
            if self.debug:
                print(f"[PocketTTSGpuEngine] Synthesis completed in {time.perf_counter() - start_time:.3f}s")
            return True
        except Exception as exc:
            traceback.print_exc()
            print(f"[PocketTTSGpuEngine] Error during synthesis: {exc}")
            return False

    def get_voices(self) -> list[PocketTTSGpuVoice]:
        return [PocketTTSGpuVoice(name) for name in PocketTTSGpuVoice.BUILTIN_VOICES]

    def set_voice(self, voice: Union[str, PocketTTSGpuVoice]) -> None:
        if isinstance(voice, str):
            if voice in PocketTTSGpuVoice.BUILTIN_VOICES:
                self.current_voice = PocketTTSGpuVoice(voice)
            elif os.path.exists(voice):
                if voice.lower().endswith(".safetensors"):
                    self.current_voice = PocketTTSGpuVoice(Path(voice).stem, state_path=voice)
                else:
                    self.current_voice = PocketTTSGpuVoice(Path(voice).stem, audio_prompt_path=voice)
            else:
                raise ValueError(
                    f"Unknown voice: {voice}. Available voices: {PocketTTSGpuVoice.BUILTIN_VOICES}"
                )
        else:
            self.current_voice = voice
        self.current_voice_state = self._get_voice_state(self.current_voice)

    def set_voice_parameters(self, **voice_parameters: Any) -> None:
        for name in ("frames_after_eos", "teacher_forcing"):
            if name in voice_parameters:
                setattr(self, name, voice_parameters[name])

    def shutdown(self) -> None:
        self._voice_states.clear()
        self.current_voice_state = None
        self.model = None
