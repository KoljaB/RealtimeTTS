"""Native Qwen3-TTS engine backed by qwentts.cpp."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from .base_engine import BaseEngine


SAMPLE_RATE = 24_000
REQUIRED_QWENTTS_ABI = 5
VOICE_CACHE_FORMAT = 1
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_QUANT = "Q8_0"
DEFAULT_ONSET_SILENCE_PROFILE = "off"
MODEL_TYPE_BASE = "base"
MODEL_TYPE_CUSTOM_VOICE = "custom_voice"
MODEL_TYPE_VOICE_DESIGN = "voice_design"
QWEN_MODEL_TYPES = frozenset(
    {MODEL_TYPE_BASE, MODEL_TYPE_CUSTOM_VOICE, MODEL_TYPE_VOICE_DESIGN}
)


def _normalize_quant(quant: str) -> str:
    aliases = {
        "F32": "F32",
        "FP32": "F32",
        "BF16": "BF16",
        "Q8": "Q8_0",
        "Q8_0": "Q8_0",
        "Q4": "Q4_K_M",
        "Q4_K_M": "Q4_K_M",
    }
    value = str(quant).strip().upper()
    if value not in aliases:
        raise ValueError(f"Unsupported Qwen quantization {quant!r}: {sorted(aliases)}")
    return aliases[value]


class QwenEngineError(RuntimeError):
    """An actionable error raised by the native Qwen engine."""


class _PauseableCancelEvent:
    """Event-compatible native control with acknowledged pause checkpoints.

    ``qwentts_cpp`` calls :meth:`is_set` from its native synthesis thread.  A
    requested pause deliberately blocks that callback until resume or cancel,
    so the acknowledgement means native generation has actually reached a
    quiescent checkpoint instead of merely buffering audio elsewhere.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._cancelled = False
        self._pause_requested = False
        self._pause_generation = 0
        self._acknowledged_generation = 0

    def cancelled(self) -> bool:
        with self._condition:
            return self._cancelled

    def is_set(self) -> bool:
        with self._condition:
            while self._pause_requested and not self._cancelled:
                self._acknowledged_generation = self._pause_generation
                self._condition.notify_all()
                self._condition.wait()
            return self._cancelled

    def set(self) -> None:
        with self._condition:
            self._cancelled = True
            self._pause_requested = False
            self._condition.notify_all()

    def request_pause(self) -> Optional[int]:
        with self._condition:
            if self._cancelled:
                return None
            if not self._pause_requested:
                self._pause_requested = True
                self._pause_generation += 1
            self._condition.notify_all()
            return self._pause_generation

    def pause(self, timeout: Optional[float] = None) -> bool:
        generation = self.request_pause()
        if generation is None:
            return False
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while (
                self._acknowledged_generation < generation
                and not self._cancelled
            ):
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return self._acknowledged_generation >= generation

    def resume(self) -> None:
        with self._condition:
            self._pause_requested = False
            self._condition.notify_all()


@dataclass
class QwenVoice:
    """Voice-cloning input for :class:`QwenEngine`.

    ``ref_text`` selects full in-context-learning (ICL) cloning. Without it,
    only the speaker embedding (x-vector) is used. Existing native ``.spk``
    and ``.rvq`` files can be supplied instead of a reference WAV.
    """

    name: str
    ref_audio: Optional[Union[str, os.PathLike[str]]] = None
    ref_text: Optional[str] = None
    language: str = "english"
    instruct: Optional[str] = None
    spk_path: Optional[Union[str, os.PathLike[str]]] = None
    rvq_path: Optional[Union[str, os.PathLike[str]]] = None
    speaker: Optional[str] = None

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.ref_text = self.ref_text.strip() if self.ref_text else None
        self.language = str(self.language).strip().lower()
        self.instruct = self.instruct.strip() if self.instruct else None
        self.speaker = self.speaker.strip() if self.speaker else None
        if not self.name:
            raise ValueError("QwenVoice.name must not be empty")
        if not self.language:
            raise ValueError("QwenVoice.language must not be empty")
        if bool(self.spk_path) != bool(self.rvq_path):
            raise ValueError("spk_path and rvq_path must be supplied together")
        if self.speaker and (self.ref_audio or self.spk_path):
            raise ValueError("speaker cannot be combined with a reference audio or cached reference")
        if not self.ref_audio and not self.spk_path and not self.speaker and not self.instruct:
            raise ValueError(
                "QwenVoice requires ref_audio, a pre-encoded spk_path/rvq_path pair, "
                "a built-in speaker, or an instruction"
            )

    @property
    def clone_mode(self) -> str:
        if self.speaker:
            return MODEL_TYPE_CUSTOM_VOICE
        if self.instruct and not self.ref_audio and not self.spk_path:
            return MODEL_TYPE_VOICE_DESIGN
        return "icl" if self.ref_text else "x_vector"

    def __repr__(self) -> str:
        return (
            f"QwenVoice(name={self.name!r}, language={self.language!r}, "
            f"clone_mode={self.clone_mode!r}, speaker={self.speaker!r})"
        )


def _infer_model_type(*values: Any) -> str:
    """Infer the native checkpoint variant from a model id or GGUF path.

    qwentts.cpp reads ``qwen3-tts.model_type`` from the GGUF, but the current
    Python ABI does not expose that metadata directly.  The model id/path is
    the same variant-qualified source used to resolve the GGUF, so use it as
    a conservative fallback until the ABI grows a metadata accessor.
    """

    text = " ".join(str(value).lower().replace("-", "_") for value in values if value)
    if "customvoice" in text or "custom_voice" in text:
        return MODEL_TYPE_CUSTOM_VOICE
    if "voicedesign" in text or "voice_design" in text:
        return MODEL_TYPE_VOICE_DESIGN
    return MODEL_TYPE_BASE


def _default_voice_cache_dir() -> Path:
    if platform.system() == "Windows":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "RealtimeTTS" / "qwen" / "voices"
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "realtimetts" / "qwen" / "voices"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _flush_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


@contextmanager
def _exclusive_cache_lock(path: Path, timeout_s: float = 600.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()} time={time.time()}\n".encode("ascii"))
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > 1800:
                    path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Qwen voice cache lock: {path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def _load_reference_audio(path: Union[str, os.PathLike[str]]) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - guarded by the installation extra
        raise ImportError(
            "QwenEngine voice extraction requires soundfile. Install "
            "with: pip install \"realtimetts[qwen]\""
        ) from exc

    audio_path = Path(path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Qwen reference audio does not exist: {audio_path}")
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"Qwen reference audio is empty: {audio_path}")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if int(sample_rate) != SAMPLE_RATE:
        try:
            import resampy
        except ImportError as exc:  # pragma: no cover - a RealtimeTTS core dependency
            raise ImportError("Resampling Qwen reference audio requires resampy") from exc
        mono = resampy.resample(mono, int(sample_rate), SAMPLE_RATE).astype(np.float32)
    mono = np.nan_to_num(mono, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.ascontiguousarray(np.clip(mono, -1.0, 1.0), dtype=np.float32)


def _float_to_pcm16(samples: np.ndarray) -> bytes:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return b""
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    audio = np.clip(audio, -1.0, 1.0)
    return np.rint(audio * 32767.0).astype("<i2", copy=False).tobytes()


class QwenEngine(BaseEngine):
    """Qwen3-TTS streaming through an in-process qwentts.cpp context."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        quant: str = DEFAULT_QUANT,
        talker_path: Optional[Union[str, os.PathLike[str]]] = None,
        codec_path: Optional[Union[str, os.PathLike[str]]] = None,
        voice: Optional[QwenVoice] = None,
        model_cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
        voice_cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
        local_files_only: bool = False,
        library_path: Optional[Union[str, os.PathLike[str]]] = None,
        use_fa: bool = True,
        clamp_fp16: bool = True,
        max_batch: int = 1,
        codec_chunk_sec: float = 24.0,
        seed: int = -1,
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        subtalker_do_sample: Optional[bool] = None,
        subtalker_temperature: Optional[float] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
        warmup: bool = True,
        warmup_text: str = "Warm up the speech engine.",
        warmup_tokens: int = 16,
        trim_silence: bool = True,
        silence_threshold: float = 0.005,
        trim_pre_roll_ms: float = 15.0,
        trim_fade_in_ms: float = 15.0,
        fragment_fade_out_after_ms: float = 1000.0,
        startup_buffer_ms: float = 160.0,
        onset_silence_profile: str = DEFAULT_ONSET_SILENCE_PROFILE,
        clone_mode: str = "auto",
        backend_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__()
        if max_batch != 1:
            raise ValueError("QwenEngine currently requires max_batch=1")
        if codec_chunk_sec <= 0:
            raise ValueError("codec_chunk_sec must be positive")
        if max_new_tokens <= 0 or warmup_tokens <= 0:
            raise ValueError("max_new_tokens and warmup_tokens must be positive")
        if silence_threshold < 0:
            raise ValueError("silence_threshold must not be negative")
        if (
            trim_pre_roll_ms < 0
            or trim_fade_in_ms < 0
            or fragment_fade_out_after_ms < 0
            or startup_buffer_ms < 0
        ):
            raise ValueError(
                "silence trimming, fragment fade, and startup buffer values must not be negative"
            )
        if bool(talker_path) != bool(codec_path):
            raise ValueError("talker_path and codec_path must be supplied together")
        if clone_mode not in {"auto", "speaker_only"}:
            raise ValueError("clone_mode must be 'auto' or 'speaker_only'")

        self.model_id = model_id
        self.quant = _normalize_quant(quant)
        self.talker_path = Path(talker_path).expanduser().resolve() if talker_path else None
        self.codec_path = Path(codec_path).expanduser().resolve() if codec_path else None
        if self.talker_path and (
            not self.talker_path.is_file() or not self.codec_path or not self.codec_path.is_file()
        ):
            raise FileNotFoundError(
                f"Qwen GGUF files not found: {self.talker_path}, {self.codec_path}"
            )
        self._explicit_model_identity: Optional[dict[str, Any]] = None
        self.model_cache_dir = Path(model_cache_dir).expanduser() if model_cache_dir else None
        self.voice_cache_dir = (
            Path(voice_cache_dir).expanduser() if voice_cache_dir else _default_voice_cache_dir()
        )
        self.local_files_only = bool(local_files_only)
        self.library_path = Path(library_path).expanduser() if library_path else None
        self.use_fa = bool(use_fa)
        self.clamp_fp16 = bool(clamp_fp16)
        self.max_batch = int(max_batch)
        self.codec_chunk_sec = float(codec_chunk_sec)
        self._warmup_enabled = bool(warmup)
        self._warmup_text = warmup_text
        self._warmup_tokens = int(warmup_tokens)
        self.trim_silence = bool(trim_silence)
        self.silence_threshold = float(silence_threshold)
        self.trim_pre_roll_ms = float(trim_pre_roll_ms)
        self.trim_fade_in_ms = float(trim_fade_in_ms)
        self.fragment_fade_out_after_ms = float(fragment_fade_out_after_ms)
        self.startup_buffer_ms = float(startup_buffer_ms)
        self.onset_silence_profile = str(
            onset_silence_profile or DEFAULT_ONSET_SILENCE_PROFILE
        ).strip().lower()
        self.clone_mode = clone_mode

        self.seed = int(seed)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.subtalker_do_sample = subtalker_do_sample
        self.subtalker_temperature = subtalker_temperature
        self.subtalker_top_k = subtalker_top_k
        self.subtalker_top_p = subtalker_top_p

        self.current_voice: Optional[QwenVoice] = None
        self._voice_ref: Any = None
        self._voice_cache_key: Optional[str] = None
        self._prepared_voice_refs: dict[str, Any] = {}
        self._warmed_voice_keys: set[str] = set()
        self._synthesis_lock = threading.RLock()
        self._control_condition = threading.Condition()
        self._pause_requested = False
        self._active_cancel_event: Optional[threading.Event] = None
        self._shutdown = False
        self.last_error: Optional[BaseException] = None
        self.last_synthesis_profile: dict[str, Any] = {}

        try:
            self._backend, binding_abi, binding_version = self._create_backend(backend_factory)
        except BaseException as exc:
            raise self._translate_error(exc, "loading the native model") from exc
        self.native_abi_version = int(binding_abi)
        self.binding_version = str(binding_version)
        self.native_version = self._native_version()
        self.model_type = self._native_model_type()
        self.instruction_control = self._supports_instruction_control()
        self.speaker_names = self._native_speaker_names()
        self.built_in_speakers = tuple(self.speaker_names)
        if self.native_abi_version != REQUIRED_QWENTTS_ABI:
            self._backend.close()
            raise QwenEngineError(
                f"qwentts.cpp ABI {self.native_abi_version} is incompatible; ABI "
                f"{REQUIRED_QWENTTS_ABI} is required. Reinstall realtimetts[qwen] "
                "and run `python -m qwentts_cpp doctor`."
            )
        if self.onset_silence_profile != DEFAULT_ONSET_SILENCE_PROFILE:
            validator = getattr(
                self._backend, "validate_onset_silence_profile", None
            )
            if not callable(validator):
                self._backend.close()
                raise QwenEngineError(
                    "The installed native Qwen binding does not support onset "
                    "silence profiles. Reinstall realtimetts[qwen]."
                )
            try:
                self.onset_silence_profile = str(
                    validator(self.onset_silence_profile)
                )
            except BaseException as exc:
                self._backend.close()
                raise self._translate_error(
                    exc, "validating the onset silence profile"
                ) from exc
        try:
            self._validate_additional_onset_silence_profiles()
        except BaseException as exc:
            try:
                self._backend.close()
            finally:
                self._backend = None
                self._shutdown = True
            raise self._translate_error(
                exc, "validating additional onset silence profiles"
            ) from exc
        if voice is not None:
            try:
                self.set_voice(voice)
            except BaseException:
                self._backend.close()
                self._backend = None
                self._shutdown = True
                raise

    def post_init(self) -> None:
        self.engine_name = "qwen"

    def _backend_options(self) -> dict[str, Any]:
        return {}

    def _load_native_module(self):
        import qwentts_cpp
        return qwentts_cpp

    def _create_backend(
        self, backend_factory: Optional[Callable[..., Any]]
    ) -> tuple[Any, int, str]:
        kwargs = {
            "model_id": self.model_id,
            "quant": self.quant,
            "talker_path": self.talker_path,
            "codec_path": self.codec_path,
            "cache_dir": self.model_cache_dir,
            "local_files_only": self.local_files_only,
            "library_path": self.library_path,
            "use_fa": self.use_fa,
            "clamp_fp16": self.clamp_fp16,
            "max_batch": self.max_batch,
            "codec_chunk_sec": self.codec_chunk_sec,
            **self._backend_options(),
        }
        if backend_factory is not None:
            backend = backend_factory(**kwargs)
            return backend, REQUIRED_QWENTTS_ABI, "injected"
        try:
            qwentts_cpp = self._load_native_module()
        except ImportError as exc:
            raise ImportError(
                "QwenEngine requires the native qwentts-cpp-python wheel. "
                "Install it with: pip install \"realtimetts[qwen]\""
            ) from exc
        abi = int(getattr(qwentts_cpp, "QT_ABI_VERSION", 0))
        if abi != REQUIRED_QWENTTS_ABI:
            raise QwenEngineError(
                f"qwentts-cpp-python exposes ABI {abi}; ABI {REQUIRED_QWENTTS_ABI} "
                "is required. Install the pinned realtimetts[qwen] dependencies."
            )
        version = getattr(qwentts_cpp, "__version__", None)
        if version is None:
            try:
                version = importlib.metadata.version("qwentts-cpp-python")
            except importlib.metadata.PackageNotFoundError:
                version = "unknown"
        if self.talker_path and self.codec_path:
            backend = qwentts_cpp.QwenTTS(
                self.talker_path,
                self.codec_path,
                library_path=self.library_path,
                use_fa=self.use_fa,
                clamp_fp16=self.clamp_fp16,
                max_batch=self.max_batch,
                codec_chunk_sec=self.codec_chunk_sec,
                **self._backend_options(),
            )
        else:
            pretrained_kwargs = dict(kwargs)
            pretrained_kwargs.pop("talker_path")
            pretrained_kwargs.pop("codec_path")
            backend = qwentts_cpp.QwenTTS.from_pretrained(**pretrained_kwargs)
        return backend, abi, str(version)

    def _native_version(self) -> str:
        try:
            version = self._backend.library.version
            return str(version() if callable(version) else version)
        except Exception:
            return "unknown"

    def _native_model_type(self) -> str:
        """Return the loaded checkpoint variant exposed by the backend.

        Newer bindings may expose ``model_type`` directly from GGUF metadata.
        ABI-5 bindings used by the current CPU deployment do not, so the
        variant-qualified model id/path remains the deterministic fallback.
        """

        for source in (self._backend, getattr(self._backend, "library", None)):
            value = getattr(source, "model_type", None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            normalized = str(value).strip().lower() if value else ""
            if normalized in QWEN_MODEL_TYPES:
                return normalized
        return _infer_model_type(self.model_id, self.talker_path)

    def _native_speaker_names(self) -> list[str]:
        if self.model_type != MODEL_TYPE_CUSTOM_VOICE:
            return []
        getter = getattr(self._backend, "speaker_names", None)
        if not callable(getter):
            return []
        try:
            return [str(name) for name in getter() if str(name).strip()]
        except Exception as exc:
            logging.debug("Unable to enumerate native Qwen speakers: %s", exc)
            return []

    def _supports_instruction_control(self) -> bool:
        if self.model_type == MODEL_TYPE_VOICE_DESIGN:
            return True
        if self.model_type != MODEL_TYPE_CUSTOM_VOICE:
            return False
        model_text = " ".join(
            str(value).lower().replace("-", "_")
            for value in (self.model_id, self.talker_path)
            if value
        )
        # The 0.6B CustomVoice checkpoint has named speakers but no
        # instruction control.  CustomVoice instructions are supported by
        # the 1.7B variant and by VoiceDesign.
        return "1.7b" in model_text or "1_7b" in model_text

    def _translate_error(self, exc: BaseException, action: str) -> BaseException:
        if isinstance(exc, ImportError):
            return exc
        message = str(exc).strip() or exc.__class__.__name__
        lowered = message.lower()
        prefix = f"QwenEngine failed while {action}: "
        if "out of memory" in lowered or "oom" in lowered:
            detail = "not enough GPU VRAM; close other GPU workloads or use a smaller quantization"
        elif "abi" in lowered or "version mismatch" in lowered:
            detail = (
                "native ABI mismatch; reinstall matching RealtimeTTS and qwentts-cpp-python wheels, "
                "then run `python -m qwentts_cpp doctor`"
            )
        elif any(token in lowered for token in ("dll", "shared libr", "libqwen", "could not find")):
            detail = (
                "a native qwen/CUDA library could not be loaded; run "
                "`python -m qwentts_cpp doctor` and reinstall realtimetts[qwen]"
            )
        elif "driver" in lowered or "no kernel image" in lowered or "cuda" in lowered:
            detail = (
                "CUDA initialization failed; verify an NVIDIA GPU with compute capability >=7.5 "
                "and a CUDA-12-compatible driver using `python -m qwentts_cpp doctor`"
            )
        else:
            detail = message
        return QwenEngineError(prefix + detail + (f" (native error: {message})" if detail != message else ""))

    def _cache_identity(self, voice: QwenVoice) -> dict[str, Any]:
        if not voice.ref_audio:
            raise ValueError("A reference WAV is required to generate a native voice cache")
        audio_path = Path(voice.ref_audio).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Qwen reference audio does not exist: {audio_path}")
        return {
            "format": VOICE_CACHE_FORMAT,
            "audio_sha256": _hash_file(audio_path),
            "ref_text": voice.ref_text or "",
            "model_id": self.model_id,
            "quant": self.quant,
            "native_abi": self.native_abi_version,
            "native_version": self.native_version,
            "model_source": self._model_source_identity(),
        }

    def _model_source_identity(self) -> dict[str, Any]:
        if not self.talker_path or not self.codec_path:
            return {"kind": "huggingface", "model_id": self.model_id, "quant": self.quant}
        if self._explicit_model_identity is None:
            self._explicit_model_identity = {
                "kind": "explicit_gguf",
                "talker_sha256": _hash_file(self.talker_path),
                "codec_sha256": _hash_file(self.codec_path),
            }
        return self._explicit_model_identity

    def _non_reference_voice_key(self, voice: QwenVoice) -> str:
        identity = {
            "format": VOICE_CACHE_FORMAT,
            "model_type": self.model_type,
            "model_id": self.model_id,
            "quant": self.quant,
            "language": voice.language,
            "speaker": voice.speaker or "",
            "instruct": voice.instruct or "",
            "model_source": self._model_source_identity(),
        }
        return hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def _cache_paths(self, identity: dict[str, Any]) -> tuple[str, Path, Path, Path]:
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        directory = self.voice_cache_dir / key[:2]
        return key, directory / f"{key}.spk", directory / f"{key}.rvq", directory / f"{key}.json"

    def _load_cached_voice(
        self, key: str, identity: dict[str, Any], spk_path: Path, rvq_path: Path,
        metadata_path: Path
    ) -> Any:
        if not (spk_path.is_file() and rvq_path.is_file() and metadata_path.is_file()):
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("cache_key") != key:
                return None
            for field, expected in identity.items():
                if metadata.get(field) != expected:
                    return None
            if metadata.get("spk_sha256") != _hash_file(spk_path):
                return None
            if metadata.get("rvq_sha256") != _hash_file(rvq_path):
                return None
            return self._backend.load_voice_ref(spk_path, rvq_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _write_voice_cache(
        self, key: str, identity: dict[str, Any], voice_ref: Any, spk_path: Path,
        rvq_path: Path, metadata_path: Path
    ) -> None:
        spk_path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        temporary_spk = spk_path.with_name(f".{spk_path.name}.{token}.tmp")
        temporary_rvq = rvq_path.with_name(f".{rvq_path.name}.{token}.tmp")
        try:
            voice_ref.save(temporary_spk, temporary_rvq)
            _flush_file(temporary_spk)
            _flush_file(temporary_rvq)
            os.replace(temporary_spk, spk_path)
            os.replace(temporary_rvq, rvq_path)
            _atomic_json(
                metadata_path,
                {
                    **identity,
                    "cache_key": key,
                    "spk_sha256": _hash_file(spk_path),
                    "rvq_sha256": _hash_file(rvq_path),
                },
            )
        finally:
            temporary_spk.unlink(missing_ok=True)
            temporary_rvq.unlink(missing_ok=True)

    def _prepare_voice(self, voice: QwenVoice) -> tuple[Optional[Any], str]:
        if self.model_type == MODEL_TYPE_CUSTOM_VOICE:
            if voice.ref_audio or voice.spk_path or voice.rvq_path:
                raise ValueError("custom_voice models use a built-in speaker, not reference audio")
            if not voice.speaker:
                raise ValueError("custom_voice models require QwenVoice(speaker=...)")
            if voice.instruct and not self.instruction_control:
                raise ValueError("This CustomVoice checkpoint does not support voice instructions")
            if self.speaker_names and voice.speaker not in self.speaker_names:
                raise ValueError(
                    f"Unknown custom_voice speaker {voice.speaker!r}; expected one of "
                    f"{self.speaker_names}"
                )
            return None, self._non_reference_voice_key(voice)

        if self.model_type == MODEL_TYPE_VOICE_DESIGN:
            if voice.ref_audio or voice.spk_path or voice.rvq_path or voice.speaker:
                raise ValueError("voice_design models use an instruction and no reference voice")
            if not voice.instruct:
                raise ValueError("voice_design models require QwenVoice(instruct=...)")
            return None, self._non_reference_voice_key(voice)

        if voice.speaker:
            raise ValueError("Base Qwen models do not support built-in speakers")
        if voice.instruct:
            raise ValueError("Base Qwen models do not support voice instructions")
        if voice.spk_path and voice.rvq_path:
            spk_path = Path(voice.spk_path).expanduser().resolve()
            rvq_path = Path(voice.rvq_path).expanduser().resolve()
            if not spk_path.is_file() or not rvq_path.is_file():
                raise FileNotFoundError(
                    f"Native Qwen voice files not found: {spk_path}, {rvq_path}"
                )
            key_data = {
                "spk_sha256": _hash_file(spk_path),
                "rvq_sha256": _hash_file(rvq_path),
                "ref_text": voice.ref_text or "",
                "model_id": self.model_id,
                "quant": self.quant,
                "native_abi": self.native_abi_version,
                "native_version": self.native_version,
                "model_source": self._model_source_identity(),
            }
            key = hashlib.sha256(
                json.dumps(key_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            cached = self._prepared_voice_refs.get(key)
            if cached is None:
                cached = self._backend.load_voice_ref(spk_path, rvq_path)
                self._prepared_voice_refs[key] = cached
            return cached, key

        identity = self._cache_identity(voice)
        key, spk_path, rvq_path, metadata_path = self._cache_paths(identity)
        prepared = self._prepared_voice_refs.get(key)
        if prepared is not None:
            return prepared, key
        with _exclusive_cache_lock(metadata_path.with_suffix(".lock")):
            cached = self._load_cached_voice(key, identity, spk_path, rvq_path, metadata_path)
            if cached is not None:
                self._prepared_voice_refs[key] = cached
                return cached, key
            audio = _load_reference_audio(voice.ref_audio)  # type: ignore[arg-type]
            voice_ref = self._backend.extract_voice_ref(audio)
            self._write_voice_cache(key, identity, voice_ref, spk_path, rvq_path, metadata_path)
            self._prepared_voice_refs[key] = voice_ref
            return voice_ref, key

    def _stream_kwargs(
        self, text: str, voice: QwenVoice, *, max_new_tokens: Optional[int] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> dict[str, Any]:
        use_icl = (
            self.model_type == MODEL_TYPE_BASE
            and voice.clone_mode == "icl"
            and self.clone_mode == "auto"
        )
        ref_spk_emb = None
        ref_codes = None
        ref_text = None
        speaker = None
        if self.model_type == MODEL_TYPE_BASE:
            if self._voice_ref is None:
                raise QwenEngineError("Base Qwen synthesis requires a prepared voice reference")
            ref_spk_emb = self._voice_ref.ref_spk_emb
            ref_codes = self._voice_ref.ref_codes if use_icl else None
            ref_text = voice.ref_text if use_icl else None
        elif self.model_type == MODEL_TYPE_CUSTOM_VOICE:
            speaker = voice.speaker
        return {
            "text": text,
            "lang": voice.language,
            "instruct": voice.instruct,
            "speaker": speaker,
            "ref_spk_emb": ref_spk_emb,
            "ref_codes": ref_codes,
            "ref_text": ref_text,
            "onset_silence_profile": (
                DEFAULT_ONSET_SILENCE_PROFILE
                if use_icl
                else self.onset_silence_profile
            ),
            "seed": self.seed,
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "subtalker_do_sample": self.subtalker_do_sample,
            "subtalker_temperature": self.subtalker_temperature,
            "subtalker_top_k": self.subtalker_top_k,
            "subtalker_top_p": self.subtalker_top_p,
            "cancel_event": cancel_event,
        }

    def _stream_native(
        self,
        stream_kwargs: dict[str, Any],
        *,
        stream_started_ns: Optional[int] = None,
    ) -> Any:
        """Create one native stream iterator.

        The CPU engine overrides this protected seam for its bounded onset
        recovery. Keeping the call here makes the normal native path exactly
        the same backend call and leaves the synthesis loop responsible for
        queueing, trimming, pause, and cancellation semantics.
        """

        del stream_started_ns
        return self._backend.stream(**stream_kwargs)

    def _native_stream_profile(self) -> dict[str, Any]:
        """Return the native profile for the primary stream attempt."""

        return dict(getattr(self._backend, "last_stream_profile", None) or {})

    def _validate_additional_onset_silence_profiles(self) -> None:
        """Hook for engines that need to validate an extra native profile."""

        return None

    def _native_stream_recovery_profile(self) -> Optional[dict[str, Any]]:
        """Return optional structured stream-recovery data for profiling."""

        return None

    def warmup(self) -> None:
        with self._synthesis_lock:
            if self.current_voice is None or self._voice_cache_key is None:
                raise QwenEngineError("Set a QwenVoice before warmup")
            warmup_key = f"{self._voice_cache_key}:{self.clone_mode}"
            if warmup_key in self._warmed_voice_keys:
                return
            cancel_event = threading.Event()
            with self._control_condition:
                self._active_cancel_event = cancel_event
                self._control_condition.notify_all()
            try:
                for _chunk, sample_rate in self._backend.stream(
                    **self._stream_kwargs(
                        self._warmup_text,
                        self.current_voice,
                        max_new_tokens=self._warmup_tokens,
                        cancel_event=cancel_event,
                    )
                ):
                    if int(sample_rate) != SAMPLE_RATE:
                        raise QwenEngineError(
                            f"qwentts.cpp returned {sample_rate} Hz during warmup; expected {SAMPLE_RATE} Hz"
                        )
                if not cancel_event.is_set():
                    self._warmed_voice_keys.add(warmup_key)
            except BaseException as exc:
                if not cancel_event.is_set():
                    raise self._translate_error(exc, "warming the native model") from exc
            finally:
                cancel_event.set()
                with self._control_condition:
                    if self._active_cancel_event is cancel_event:
                        self._active_cancel_event = None
                    self._control_condition.notify_all()

    def get_stream_info(self) -> Tuple[int, int, int]:
        from .._audio_backend import pyaudio

        return pyaudio.paInt16, 1, SAMPLE_RATE

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)
        if not isinstance(text, str) or not text.strip():
            self.last_error = ValueError("QwenEngine text must not be empty")
            return False
        started_ns = time.perf_counter_ns()
        cancel_event = _PauseableCancelEvent()
        with self._synthesis_lock:
            if self._shutdown or self._backend is None:
                self.last_error = QwenEngineError("QwenEngine is shut down")
                return False
            if self.current_voice is None:
                self.last_error = QwenEngineError("Set a QwenVoice before synthesis")
                return False
            with self._control_condition:
                self._active_cancel_event = cancel_event
                if self._pause_requested:
                    cancel_event.request_pause()
                self._control_condition.notify_all()
            self.last_error = None
            first_queue_ns: Optional[int] = None
            n_samples = 0
            native_samples = 0
            native_peak = 0.0
            native_overrange_samples = 0
            native_max_sample_jump = 0.0
            previous_native_sample: Optional[float] = None
            queued_chunks: list[dict[str, Any]] = []
            startup_audio: list[np.ndarray] = []
            startup_samples = 0
            startup_emitted = False
            pre_roll_samples = int(round(self.trim_pre_roll_ms * SAMPLE_RATE / 1000))
            fade_in_samples = int(round(self.trim_fade_in_ms * SAMPLE_RATE / 1000))
            ending_fade_samples = fade_in_samples if self.trim_silence else 0
            ending_fade_after_samples = int(
                round(self.fragment_fade_out_after_ms * SAMPLE_RATE / 1000)
            )
            startup_target_samples = int(round(self.startup_buffer_ms * SAMPLE_RATE / 1000))
            trim_window_samples = self._silence_trim_window_samples(SAMPLE_RATE)
            quiet_tail = np.empty(0, dtype=np.float32)
            search_remainder = np.empty(0, dtype=np.float32)
            leading_trimmed_samples = 0
            startup_fade_samples = 0
            ending_fade_applied_samples = 0
            ending_audio = np.empty(0, dtype=np.float32)
            ending_fade_input_samples = 0
            callback_profile: dict[str, Any] = {}
            stream = None

            def publish_audio(audio: np.ndarray) -> None:
                nonlocal first_queue_ns, n_samples
                pcm = _float_to_pcm16(audio)
                if not pcm:
                    return
                chunk_samples = len(pcm) // 2
                self.queue.put(pcm)
                queued_ns = time.perf_counter_ns()
                if first_queue_ns is None:
                    first_queue_ns = queued_ns
                    margin_before_ms = None
                else:
                    elapsed_ms = (queued_ns - first_queue_ns) / 1_000_000
                    margin_before_ms = n_samples * 1000 / SAMPLE_RATE - elapsed_ms
                queued_chunks.append(
                    {
                        "queued_ms": (queued_ns - started_ns) / 1_000_000,
                        "samples": chunk_samples,
                        "duration_ms": chunk_samples * 1000 / SAMPLE_RATE,
                        "playout_margin_before_ms": margin_before_ms,
                    }
                )
                n_samples += chunk_samples

            def publish_with_ending_fade(audio: np.ndarray) -> None:
                nonlocal ending_audio, ending_fade_input_samples
                if audio.size == 0:
                    return
                if ending_fade_samples <= 0:
                    publish_audio(audio)
                    return
                if ending_fade_input_samples < ending_fade_after_samples:
                    immediate_samples = min(
                        int(audio.size),
                        ending_fade_after_samples - ending_fade_input_samples,
                    )
                    if immediate_samples:
                        publish_audio(audio[:immediate_samples])
                        ending_fade_input_samples += immediate_samples
                        audio = audio[immediate_samples:]
                    if audio.size == 0:
                        return
                combined = (
                    audio
                    if ending_audio.size == 0
                    else np.concatenate((ending_audio, audio))
                )
                if combined.size <= ending_fade_samples:
                    ending_audio = combined
                    return
                split = int(combined.size) - ending_fade_samples
                publish_audio(combined[:split])
                ending_audio = combined[split:].copy()

            def publish_startup_audio(audio: np.ndarray) -> None:
                nonlocal startup_fade_samples
                if (
                    self.trim_silence
                    and startup_fade_samples == 0
                    and fade_in_samples > 0
                    and audio.size > 0
                ):
                    startup_fade_samples = min(fade_in_samples, int(audio.size))
                    audio = self.apply_fade_in(
                        audio,
                        SAMPLE_RATE,
                        int(round(self.trim_fade_in_ms)),
                    )
                publish_with_ending_fade(audio)

            def buffer_or_publish(audio: np.ndarray) -> None:
                nonlocal startup_samples, startup_emitted
                if audio.size == 0:
                    return
                if startup_emitted:
                    publish_with_ending_fade(audio)
                    return
                startup_audio.append(audio)
                startup_samples += int(audio.size)
                if startup_target_samples == 0 or startup_samples >= startup_target_samples:
                    publish_startup_audio(np.concatenate(startup_audio))
                    startup_audio.clear()
                    startup_samples = 0
                    startup_emitted = True

            def detect_start(audio: np.ndarray, *, final: bool = False) -> Optional[np.ndarray]:
                nonlocal quiet_tail, search_remainder, leading_trimmed_samples
                nonlocal startup_fade_samples
                data = np.concatenate((search_remainder, audio))
                search_remainder = np.empty(0, dtype=np.float32)
                complete_samples = (
                    int(data.size)
                    if final
                    else int(data.size) // trim_window_samples * trim_window_samples
                )
                position = 0
                while position < complete_samples:
                    end = min(position + trim_window_samples, complete_samples)
                    if self._window_is_non_silent(
                        np.abs(data[position:end]), self.silence_threshold
                    ):
                        retained_prefix = np.concatenate((quiet_tail, data[:position]))
                        retained_prefix = (
                            retained_prefix[-pre_roll_samples:]
                            if pre_roll_samples
                            else np.empty(0, dtype=np.float32)
                        )
                        result = np.concatenate((retained_prefix, data[position:]))
                        leading_trimmed_samples = native_samples - int(result.size)
                        if leading_trimmed_samples > 0 and fade_in_samples > 0:
                            startup_fade_samples = min(fade_in_samples, int(result.size))
                            result = result.copy()
                            result[:startup_fade_samples] *= np.linspace(
                                0.0, 1.0, startup_fade_samples, dtype=np.float32
                            )
                        return result
                    position = end
                classified = data[:complete_samples]
                if classified.size and pre_roll_samples:
                    quiet_tail = np.concatenate((quiet_tail, classified))[-pre_roll_samples:]
                search_remainder = data[complete_samples:]
                return None

            try:
                stream_started_ns = time.perf_counter_ns()
                stream = self._stream_native(
                    self._stream_kwargs(text.strip(), self.current_voice, cancel_event=cancel_event),
                    stream_started_ns=stream_started_ns,
                )
                for chunk, sample_rate in stream:
                    if self.stop_synthesis_event.is_set() or cancel_event.cancelled():
                        cancel_event.set()
                        break
                    if int(sample_rate) != SAMPLE_RATE:
                        raise QwenEngineError(
                            f"qwentts.cpp returned {sample_rate} Hz; expected {SAMPLE_RATE} Hz mono"
                        )
                    audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
                    if audio.size:
                        native_peak = max(native_peak, float(np.max(np.abs(audio))))
                        native_overrange_samples += int(np.count_nonzero(np.abs(audio) > 1.0))
                        if audio.size > 1:
                            native_max_sample_jump = max(
                                native_max_sample_jump,
                                float(np.max(np.abs(np.diff(audio)))),
                            )
                        if previous_native_sample is not None:
                            native_max_sample_jump = max(
                                native_max_sample_jump,
                                abs(float(audio[0]) - previous_native_sample),
                            )
                        previous_native_sample = float(audio[-1])
                    native_samples += int(audio.size)
                    if self.trim_silence and self._trim_silence_start_pending:
                        detected = detect_start(audio)
                        if detected is None:
                            continue
                        audio = detected
                        self._trim_silence_start_pending = False
                    buffer_or_publish(audio)
                if (
                    self.trim_silence
                    and self._trim_silence_start_pending
                    and search_remainder.size
                    and not (cancel_event.cancelled() or self.stop_synthesis_event.is_set())
                ):
                    detected = detect_start(np.empty(0, dtype=np.float32), final=True)
                    if detected is not None:
                        self._trim_silence_start_pending = False
                        buffer_or_publish(detected)
                if startup_audio and not (
                    cancel_event.cancelled() or self.stop_synthesis_event.is_set()
                ):
                    publish_startup_audio(np.concatenate(startup_audio))
                    startup_audio.clear()
                    startup_samples = 0
                    startup_emitted = True
                if ending_audio.size and not (
                    cancel_event.cancelled() or self.stop_synthesis_event.is_set()
                ):
                    ending_fade_applied_samples = min(
                        ending_fade_samples, int(ending_audio.size)
                    )
                    publish_audio(
                        self.apply_fade_out(
                            ending_audio,
                            SAMPLE_RATE,
                            int(round(self.trim_fade_in_ms)),
                        )
                    )
                    ending_audio = np.empty(0, dtype=np.float32)
                callback_profile = self._native_stream_profile()
                margins = [
                    item["playout_margin_before_ms"]
                    for item in queued_chunks
                    if item["playout_margin_before_ms"] is not None
                ]
                self.audio_duration += n_samples / SAMPLE_RATE
                self.last_synthesis_profile = {
                    "cancelled": cancel_event.cancelled() or self.stop_synthesis_event.is_set(),
                    "n_samples": n_samples,
                    "leading_trimmed_samples": leading_trimmed_samples,
                    "leading_trimmed_ms": leading_trimmed_samples * 1000 / SAMPLE_RATE,
                    "startup_speech_detected": (
                        not self.trim_silence or not self._trim_silence_start_pending
                    ),
                    "startup_buffered_ms": (
                        queued_chunks[0]["duration_ms"] if queued_chunks else None
                    ),
                    "ending_fade_samples": ending_fade_applied_samples,
                    "ending_fade_after_ms": self.fragment_fade_out_after_ms,
                    "startup_target_ms": self.startup_buffer_ms,
                    "startup_fade_samples": startup_fade_samples,
                    "native_peak": native_peak,
                    "native_overrange_samples": native_overrange_samples,
                    "native_max_sample_jump": native_max_sample_jump,
                    "audio_duration_s": n_samples / SAMPLE_RATE,
                    "queued_chunks": queued_chunks,
                    "first_chunk_duration_ms": (
                        queued_chunks[0]["duration_ms"] if queued_chunks else None
                    ),
                    "minimum_playout_margin_ms": min(margins) if margins else None,
                    "predicted_underruns": sum(margin < 0 for margin in margins),
                    "total_ms": (time.perf_counter_ns() - started_ns) / 1_000_000,
                    "first_queue_ms": (
                        (first_queue_ns - started_ns) / 1_000_000 if first_queue_ns else None
                    ),
                    "native": callback_profile,
                }
                recovery_profile = self._native_stream_recovery_profile()
                if recovery_profile is not None:
                    self.last_synthesis_profile["onset_recovery"] = recovery_profile
                if first_queue_ns is not None:
                    callback_ns = callback_profile.get("first_callback_perf_counter_ns")
                    if callback_ns is not None:
                        self.last_synthesis_profile["callback_to_queue_ms"] = (
                            first_queue_ns - int(callback_ns)
                        ) / 1_000_000
                    elif "first_callback_enter_ms" in callback_profile:
                        self.last_synthesis_profile["callback_to_queue_ms"] = (
                            (first_queue_ns - stream_started_ns) / 1_000_000
                            - float(callback_profile["first_callback_enter_ms"])
                        )
                if n_samples == 0 and not (
                    cancel_event.cancelled() or self.stop_synthesis_event.is_set()
                ):
                    raise QwenEngineError("qwentts.cpp completed without producing audio")
                return True
            except BaseException as exc:
                recovery_profile = self._native_stream_recovery_profile()
                if recovery_profile is not None:
                    self.last_synthesis_profile = {
                        "cancelled": cancel_event.cancelled() or self.stop_synthesis_event.is_set(),
                        "n_samples": n_samples,
                        "total_ms": (time.perf_counter_ns() - started_ns) / 1_000_000,
                        "native": self._native_stream_profile(),
                        "onset_recovery": recovery_profile,
                    }
                if cancel_event.cancelled() or self.stop_synthesis_event.is_set():
                    return True
                self.last_error = self._translate_error(exc, "streaming synthesis")
                logging.exception("QwenEngine synthesis failed: %s", self.last_error)
                return False
            finally:
                cancel_event.set()
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        logging.debug("Failed to close cancelled qwentts stream", exc_info=True)
                with self._control_condition:
                    if self._active_cancel_event is cancel_event:
                        self._active_cancel_event = None
                    self._control_condition.notify_all()

    def get_voices(self) -> list[QwenVoice]:
        if self.model_type != MODEL_TYPE_CUSTOM_VOICE:
            return []
        return [
            QwenVoice(name=name, speaker=name, language="english")
            for name in self.speaker_names
        ]

    def set_voice(self, voice: Union[str, QwenVoice]) -> None:
        if not isinstance(voice, QwenVoice):
            raise TypeError("QwenEngine.set_voice expects a QwenVoice")
        with self._synthesis_lock:
            if self._shutdown or self._backend is None:
                raise QwenEngineError("QwenEngine is shut down")
            previous = (self.current_voice, self._voice_ref, self._voice_cache_key)
            try:
                voice_ref, cache_key = self._prepare_voice(voice)
                self.current_voice = voice
                self._voice_ref = voice_ref
                self._voice_cache_key = cache_key
                if self._warmup_enabled:
                    self.warmup()
            except BaseException as exc:
                self.current_voice, self._voice_ref, self._voice_cache_key = previous
                if isinstance(exc, (ValueError, TypeError, FileNotFoundError, ImportError)):
                    raise
                raise self._translate_error(exc, f"preparing voice {voice.name!r}") from exc

    def set_voice_parameters(self, **voice_parameters: Any) -> None:
        voice_fields = {"language", "instruct", "speaker"}
        sampling_fields = {
            "seed", "max_new_tokens", "do_sample", "temperature", "top_k", "top_p",
            "repetition_penalty", "subtalker_do_sample", "subtalker_temperature",
            "subtalker_top_k", "subtalker_top_p", "clone_mode",
        }
        unknown = set(voice_parameters) - voice_fields - sampling_fields
        if unknown:
            raise ValueError(f"Unsupported QwenEngine voice parameters: {sorted(unknown)}")
        if "clone_mode" in voice_parameters and voice_parameters["clone_mode"] not in {"auto", "speaker_only"}:
            raise ValueError("clone_mode must be 'auto' or 'speaker_only'")
        with self._synthesis_lock:
            if self.current_voice:
                if "language" in voice_parameters:
                    language = str(voice_parameters.pop("language")).strip().lower()
                    if not language:
                        raise ValueError("language must not be empty")
                    self.current_voice.language = language
                if "instruct" in voice_parameters:
                    value = voice_parameters.pop("instruct")
                    if value and not self.instruction_control:
                        raise ValueError("This Qwen checkpoint does not support voice instructions")
                    self.current_voice.instruct = str(value).strip() if value else None
                if "speaker" in voice_parameters:
                    value = voice_parameters.pop("speaker")
                    speaker = str(value).strip() if value else None
                    if self.model_type == MODEL_TYPE_CUSTOM_VOICE and not speaker:
                        raise ValueError("custom_voice models require a speaker")
                    if self.speaker_names and speaker not in self.speaker_names:
                        raise ValueError(
                            f"Unknown custom_voice speaker {speaker!r}; expected one of "
                            f"{self.speaker_names}"
                        )
                    self.current_voice.speaker = speaker
            for name, value in voice_parameters.items():
                setattr(self, name, value)

    def pause(self, timeout: Optional[float] = 1.0) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._control_condition:
            self._pause_requested = True
            active = self._active_cancel_event
            while active is None and not self._shutdown:
                if deadline is None:
                    self._control_condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return False
                    self._control_condition.wait(remaining)
                active = self._active_cancel_event
        pause = getattr(active, "pause", None)
        if not callable(pause):
            return False
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        return bool(pause(timeout=remaining))

    def resume(self) -> None:
        with self._control_condition:
            self._pause_requested = False
            active = self._active_cancel_event
            self._control_condition.notify_all()
        resume = getattr(active, "resume", None)
        if callable(resume):
            resume()

    def stop(self) -> None:
        super().stop()
        with self._control_condition:
            active = self._active_cancel_event
            self._control_condition.notify_all()
        if active is not None:
            active.set()

    def shutdown(self) -> None:
        self.stop()
        with self._synthesis_lock:
            if self._shutdown:
                return
            self._shutdown = True
            backend, self._backend = self._backend, None
            self._voice_ref = None
            self._prepared_voice_refs.clear()
            self.current_voice = None
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    logging.exception("Failed to close qwentts.cpp cleanly")


__all__ = [
    "MODEL_TYPE_BASE",
    "MODEL_TYPE_CUSTOM_VOICE",
    "MODEL_TYPE_VOICE_DESIGN",
    "QwenEngine",
    "QwenEngineError",
    "QwenVoice",
]
