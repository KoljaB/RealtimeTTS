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
REQUIRED_QWENTTS_ABI = 4
VOICE_CACHE_FORMAT = 1
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_QUANT = "Q8_0"


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

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.ref_text = self.ref_text.strip() if self.ref_text else None
        self.language = str(self.language).strip().lower()
        self.instruct = self.instruct.strip() if self.instruct else None
        if not self.name:
            raise ValueError("QwenVoice.name must not be empty")
        if not self.language:
            raise ValueError("QwenVoice.language must not be empty")
        if bool(self.spk_path) != bool(self.rvq_path):
            raise ValueError("spk_path and rvq_path must be supplied together")
        if not self.ref_audio and not self.spk_path:
            raise ValueError(
                "QwenVoice requires ref_audio or a pre-encoded spk_path/rvq_path pair"
            )

    @property
    def clone_mode(self) -> str:
        return "icl" if self.ref_text else "x_vector"

    def __repr__(self) -> str:
        return (
            f"QwenVoice(name={self.name!r}, language={self.language!r}, "
            f"clone_mode={self.clone_mode!r})"
        )


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
        trim_fade_in_ms: float = 20.0,
        startup_buffer_ms: float = 160.0,
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
        if trim_pre_roll_ms < 0 or trim_fade_in_ms < 0 or startup_buffer_ms < 0:
            raise ValueError(
                "trim_pre_roll_ms, trim_fade_in_ms, and startup_buffer_ms must not be negative"
            )
        if bool(talker_path) != bool(codec_path):
            raise ValueError("talker_path and codec_path must be supplied together")

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
        self.startup_buffer_ms = float(startup_buffer_ms)

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
        if self.native_abi_version != REQUIRED_QWENTTS_ABI:
            self._backend.close()
            raise QwenEngineError(
                f"qwentts.cpp ABI {self.native_abi_version} is incompatible; ABI "
                f"{REQUIRED_QWENTTS_ABI} is required. Reinstall realtimetts[qwen] "
                "and run `python -m qwentts_cpp doctor`."
            )
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
        }
        if backend_factory is not None:
            backend = backend_factory(**kwargs)
            return backend, REQUIRED_QWENTTS_ABI, "injected"
        try:
            import qwentts_cpp
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
            "binding_version": self.binding_version,
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

    def _prepare_voice(self, voice: QwenVoice) -> tuple[Any, str]:
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
        use_icl = voice.clone_mode == "icl"
        return {
            "text": text,
            "lang": voice.language,
            "instruct": voice.instruct,
            "ref_spk_emb": self._voice_ref.ref_spk_emb,
            "ref_codes": self._voice_ref.ref_codes if use_icl else None,
            "ref_text": voice.ref_text if use_icl else None,
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

    def warmup(self) -> None:
        with self._synthesis_lock:
            if self.current_voice is None or self._voice_ref is None or self._voice_cache_key is None:
                raise QwenEngineError("Set a QwenVoice before warmup")
            if self._voice_cache_key in self._warmed_voice_keys:
                return
            cancel_event = threading.Event()
            self._active_cancel_event = cancel_event
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
                    self._warmed_voice_keys.add(self._voice_cache_key)
            except BaseException as exc:
                if not cancel_event.is_set():
                    raise self._translate_error(exc, "warming the native model") from exc
            finally:
                cancel_event.set()
                if self._active_cancel_event is cancel_event:
                    self._active_cancel_event = None

    def get_stream_info(self) -> Tuple[int, int, int]:
        from .._audio_backend import pyaudio

        return pyaudio.paInt16, 1, SAMPLE_RATE

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)
        if not isinstance(text, str) or not text.strip():
            self.last_error = ValueError("QwenEngine text must not be empty")
            return False
        started_ns = time.perf_counter_ns()
        cancel_event = threading.Event()
        with self._synthesis_lock:
            if self._shutdown or self._backend is None:
                self.last_error = QwenEngineError("QwenEngine is shut down")
                return False
            if self.current_voice is None or self._voice_ref is None:
                self.last_error = QwenEngineError("Set a QwenVoice before synthesis")
                return False
            self._active_cancel_event = cancel_event
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
            startup_target_samples = int(round(self.startup_buffer_ms * SAMPLE_RATE / 1000))
            pre_roll_samples = int(round(self.trim_pre_roll_ms * SAMPLE_RATE / 1000))
            fade_in_samples = int(round(self.trim_fade_in_ms * SAMPLE_RATE / 1000))
            trim_window_samples = self._silence_trim_window_samples(SAMPLE_RATE)
            quiet_tail = np.empty(0, dtype=np.float32)
            search_remainder = np.empty(0, dtype=np.float32)
            leading_trimmed_samples = 0
            startup_fade_samples = 0
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

            def buffer_or_publish(audio: np.ndarray) -> None:
                nonlocal startup_samples, startup_emitted
                if audio.size == 0:
                    return
                if startup_emitted:
                    publish_audio(audio)
                    return
                startup_audio.append(audio)
                startup_samples += int(audio.size)
                if startup_target_samples == 0 or startup_samples >= startup_target_samples:
                    publish_audio(np.concatenate(startup_audio))
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
                stream = self._backend.stream(
                    **self._stream_kwargs(text.strip(), self.current_voice, cancel_event=cancel_event)
                )
                for chunk, sample_rate in stream:
                    if self.stop_synthesis_event.is_set() or cancel_event.is_set():
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
                    and not (cancel_event.is_set() or self.stop_synthesis_event.is_set())
                ):
                    detected = detect_start(np.empty(0, dtype=np.float32), final=True)
                    if detected is not None:
                        self._trim_silence_start_pending = False
                        buffer_or_publish(detected)
                if startup_audio and not (
                    cancel_event.is_set() or self.stop_synthesis_event.is_set()
                ):
                    publish_audio(np.concatenate(startup_audio))
                    startup_audio.clear()
                    startup_samples = 0
                    startup_emitted = True
                callback_profile = dict(getattr(self._backend, "last_stream_profile", None) or {})
                margins = [
                    item["playout_margin_before_ms"]
                    for item in queued_chunks
                    if item["playout_margin_before_ms"] is not None
                ]
                self.audio_duration += n_samples / SAMPLE_RATE
                self.last_synthesis_profile = {
                    "cancelled": cancel_event.is_set() or self.stop_synthesis_event.is_set(),
                    "n_samples": n_samples,
                    "leading_trimmed_samples": leading_trimmed_samples,
                    "leading_trimmed_ms": leading_trimmed_samples * 1000 / SAMPLE_RATE,
                    "startup_speech_detected": (
                        not self.trim_silence or not self._trim_silence_start_pending
                    ),
                    "startup_buffered_ms": (
                        queued_chunks[0]["duration_ms"] if queued_chunks else None
                    ),
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
                    cancel_event.is_set() or self.stop_synthesis_event.is_set()
                ):
                    raise QwenEngineError("qwentts.cpp completed without producing audio")
                return True
            except BaseException as exc:
                if cancel_event.is_set() or self.stop_synthesis_event.is_set():
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
                self._active_cancel_event = None

    def get_voices(self) -> list[QwenVoice]:
        return []

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
        voice_fields = {"language", "instruct"}
        sampling_fields = {
            "seed", "max_new_tokens", "do_sample", "temperature", "top_k", "top_p",
            "repetition_penalty", "subtalker_do_sample", "subtalker_temperature",
            "subtalker_top_k", "subtalker_top_p",
        }
        unknown = set(voice_parameters) - voice_fields - sampling_fields
        if unknown:
            raise ValueError(f"Unsupported QwenEngine voice parameters: {sorted(unknown)}")
        with self._synthesis_lock:
            if self.current_voice:
                if "language" in voice_parameters:
                    language = str(voice_parameters.pop("language")).strip().lower()
                    if not language:
                        raise ValueError("language must not be empty")
                    self.current_voice.language = language
                if "instruct" in voice_parameters:
                    value = voice_parameters.pop("instruct")
                    self.current_voice.instruct = str(value).strip() if value else None
            for name, value in voice_parameters.items():
                setattr(self, name, value)

    def stop(self) -> None:
        super().stop()
        active = self._active_cancel_event
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
    "QwenEngine",
    "QwenEngineError",
    "QwenVoice",
]
