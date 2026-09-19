"""OpenAI-compatible HTTP server for :class:`QwenEngine`.

The server deliberately keeps the HTTP contract used by qwentts.cpp's native
``tts-server``.  In particular, a PCM response is not committed until the
first audible block exists.  A request that completes with only silence can
therefore still return a real JSON error instead of a misleading empty 200.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import hashlib
import importlib.metadata
import ipaddress
import io
import json
import logging
import os
import queue as queue_module
import re
import secrets
import threading
import time
import uuid
import wave
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
from stream2sentence import generate_sentences_async

try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
except ImportError:  # pragma: no cover - guarded by qwen-server extra
    FastAPI = None  # type: ignore[assignment]
    Request = Any  # type: ignore[misc,assignment]
    WebSocket = Any  # type: ignore[misc,assignment]
    WebSocketDisconnect = Exception  # type: ignore[assignment]
    CORSMiddleware = None  # type: ignore[assignment]
    JSONResponse = Response = StreamingResponse = None  # type: ignore[assignment]

from .engines.qwen_engine import (
    DEFAULT_MODEL,
    DEFAULT_QUANT,
    SAMPLE_RATE,
    QwenEngine,
    QwenVoice,
)
from .engines.qwen_cpu_engine import QwenCpuEngine
from .language_router import (
    FastTextLanguageDetector,
    LanguageDetection,
    QwenLanguageRouter,
    language_lookahead_wait_ms,
)


LOGGER = logging.getLogger(__name__)
AUDIBLE_AVERAGE_ABS_THRESHOLD = 80.0
DEFAULT_STALL_TIMEOUT_SECONDS = 30.0
DEFAULT_SYNTHESIS_TIMEOUT_SECONDS = 120.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 5.0
PAUSE_ACK_TIMEOUT_SECONDS = 2.0
DEFAULT_MAX_ACTIVE_REQUESTS = 1
DEFAULT_MAX_QUEUED_REQUESTS = 8
DEFAULT_OUTPUT_QUEUE_CHUNKS = 128
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_STREAM_EVENT_BYTES = 64 * 1024
MAX_STREAM_TEXT_BYTES = 1024 * 1024
VOICE_REGISTRY_FORMAT = 1
SERVER_PROTOCOL = "realtimetts-qwen-http"
SERVER_PROTOCOL_VERSION = 1
DEFAULT_CORS_ORIGINS = (
    "http://localhost",
    "http://127.0.0.1",
    "http://[::1]",
)
AUDIO_METADATA = {
    "encoding": "s16le",
    "sample_rate_hz": SAMPLE_RATE,
    "channels": 1,
    "bits_per_sample": 16,
}
_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_STREAM_END = object()
_TEXT_END = object()
_TEXT_FLUSH = object()


@dataclass(frozen=True)
class _SegmentBoundary:
    segment_id: int
    started: bool
_END_SENTENCE_DELIMITERS = ".!?。！？…"
_MID_SENTENCE_DELIMITERS = ";:,\n()[]{}-“”„”—/|《》"
LANGUAGE_LOOKAHEAD_MAX_ADDITIONAL_WORDS = 3
_NO_FORCED_FIRST_FRAGMENT = float("inf")
_QWEN_TO_TOKENIZER_LANGUAGE = {
    "chinese": "zh",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
    "german": "de",
    "french": "fr",
    "russian": "ru",
    "portuguese": "pt",
    "spanish": "es",
    "italian": "it",
}


class ApiError(Exception):
    def __init__(self, status: int, error_type: str, message: str) -> None:
        super().__init__(message)
        self.status = int(status)
        self.error_type = error_type
        self.message = message


def _server_version() -> str:
    try:
        # Prefer the source package: an existing installed distribution can
        # describe an older checkout when the server is run from a worktree.
        from ._version import __version__
    except (ImportError, AttributeError):
        try:
            return importlib.metadata.version("realtimetts")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"
    return str(__version__)


def _new_client_id(value: Any = None) -> str:
    if isinstance(value, str):
        candidate = value.strip()
        if _CLIENT_ID_RE.fullmatch(candidate):
            return candidate
    return uuid.uuid4().hex


def _audio_response_headers() -> dict[str, str]:
    return {
        "X-Audio-Encoding": str(AUDIO_METADATA["encoding"]),
        "X-Audio-Sample-Rate": str(AUDIO_METADATA["sample_rate_hz"]),
        "X-Audio-Channels": str(AUDIO_METADATA["channels"]),
        "X-Audio-Bits-Per-Sample": str(AUDIO_METADATA["bits_per_sample"]),
    }


def _is_loopback_host(value: str) -> bool:
    candidate = str(value).strip().lower().strip("[]")
    if candidate in {"localhost", "ip6-localhost"}:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _error_payload(error_type: str, message: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type}}


def _pcm_is_audible(pcm: bytes) -> bool:
    usable = len(pcm) - (len(pcm) % 2)
    if usable <= 0:
        return False
    samples = np.frombuffer(pcm[:usable], dtype="<i2").astype(np.int32)
    return bool(samples.size and np.mean(np.abs(samples)) >= AUDIBLE_AVERAGE_ABS_THRESHOLD)


def _pcm_to_wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return output.getvalue()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            if os.name == "posix":
                temporary.chmod(0o600)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _write_candidate(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            if os.name == "posix":
                temporary.chmod(0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _decode_base64(value: Any, field: str) -> Optional[bytes]:
    if not isinstance(value, str) or not value:
        return None
    compact = value.replace("\r", "").replace("\n", "")
    compact += "=" * (-len(compact) % 4)
    try:
        decoded = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ApiError(400, "invalid_request_error", "invalid base64 payload") from exc
    if not decoded:
        return None
    return decoded


@dataclass(frozen=True)
class VoiceRecord:
    name: str
    ref_text: str
    kind: str
    files: tuple[str, ...]
    sha256: tuple[str, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ref_text": self.ref_text,
            "kind": self.kind,
            "files": list(self.files),
            "sha256": list(self.sha256),
        }


class VoiceRegistry:
    """Small persistent registry for uploaded WAV or native voice latents."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.files_dir = self.root / "files"
        self.manifest_path = self.root / "manifest.json"
        self._lock = threading.RLock()
        self._records: dict[str, VoiceRecord] = {}
        self.files_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            self.root.chmod(0o700)
            self.files_dir.chmod(0o700)
        self._load()

    def _resolve_file(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("voice registry file escapes its root") from exc
        return candidate

    def _load(self) -> None:
        if not self.manifest_path.is_file():
            return
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if payload.get("format") != VOICE_REGISTRY_FORMAT:
                raise ValueError("unsupported registry format")
            raw_records = payload.get("voices")
            if not isinstance(raw_records, list):
                raise ValueError("registry voices must be a list")
            loaded: dict[str, VoiceRecord] = {}
            for raw in raw_records:
                if not isinstance(raw, dict):
                    raise ValueError("registry voice must be an object")
                record = VoiceRecord(
                    name=str(raw["name"]),
                    ref_text=str(raw.get("ref_text") or ""),
                    kind=str(raw["kind"]),
                    files=tuple(str(value) for value in raw["files"]),
                    sha256=tuple(str(value) for value in raw["sha256"]),
                )
                expected_files = 1 if record.kind == "wav" else 2 if record.kind == "latents" else 0
                if (
                    not record.name
                    or len(record.files) != expected_files
                    or len(record.sha256) != expected_files
                ):
                    raise ValueError(f"invalid registry record for {record.name!r}")
                for relative, expected_hash in zip(record.files, record.sha256):
                    path = self._resolve_file(relative)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                        raise ValueError(f"voice registry hash mismatch: {path}")
                loaded[record.name] = record
            self._records = loaded
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            LOGGER.exception("Ignoring invalid Qwen voice registry at %s", self.manifest_path)
            self._records = {}

    def _write_manifest(self, records: Mapping[str, VoiceRecord]) -> None:
        _atomic_json(
            self.manifest_path,
            {
                "format": VOICE_REGISTRY_FORMAT,
                "voices": [records[name].to_manifest() for name in sorted(records)],
            },
        )

    def records(self) -> list[VoiceRecord]:
        with self._lock:
            return [self._records[name] for name in sorted(self._records)]

    def get(self, name: str) -> Optional[VoiceRecord]:
        with self._lock:
            return self._records.get(name)

    def candidate(
        self, name: str, ref_text: str, kind: str, payloads: Sequence[bytes]
    ) -> VoiceRecord:
        suffixes = (".wav",) if kind == "wav" else (".spk", ".rvq")
        token = f"{hashlib.sha256(name.encode('utf-8')).hexdigest()[:16]}-{uuid.uuid4().hex}"
        relatives: list[str] = []
        hashes: list[str] = []
        try:
            for suffix, payload in zip(suffixes, payloads):
                relative = f"files/{token}{suffix}"
                _write_candidate(self._resolve_file(relative), payload)
                relatives.append(relative)
                hashes.append(hashlib.sha256(payload).hexdigest())
        except BaseException:
            for relative in relatives:
                self._resolve_file(relative).unlink(missing_ok=True)
            raise
        return VoiceRecord(name, ref_text, kind, tuple(relatives), tuple(hashes))

    def discard_candidate(self, record: VoiceRecord) -> None:
        for relative in record.files:
            self._resolve_file(relative).unlink(missing_ok=True)

    def commit(self, record: VoiceRecord) -> None:
        with self._lock:
            previous = self._records.get(record.name)
            updated = dict(self._records)
            updated[record.name] = record
            self._write_manifest(updated)
            self._records = updated
        if previous is not None:
            self.discard_candidate(previous)

    def remove(self, name: str) -> bool:
        with self._lock:
            previous = self._records.get(name)
            if previous is None:
                return False
            updated = dict(self._records)
            del updated[name]
            self._write_manifest(updated)
            self._records = updated
        self.discard_candidate(previous)
        return True

    def to_voice(
        self,
        record: VoiceRecord,
        language: str,
        instruct: Optional[str] = None,
    ) -> QwenVoice:
        paths = [self._resolve_file(relative) for relative in record.files]
        if record.kind == "wav":
            return QwenVoice(
                record.name,
                ref_audio=paths[0],
                ref_text=record.ref_text or None,
                language=language,
                instruct=instruct,
            )
        return QwenVoice(
            record.name,
            spk_path=paths[0],
            rvq_path=paths[1],
            ref_text=record.ref_text or None,
            language=language,
            instruct=instruct,
        )


class RequestMetrics:
    def __init__(
        self,
        stall_timeout_seconds: float = DEFAULT_STALL_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if stall_timeout_seconds <= 0:
            raise ValueError("stall_timeout_seconds must be positive")
        self.stall_timeout_seconds = float(stall_timeout_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._active = 0
        self._requests_total = 0
        self._synthesis_failures_total = 0
        self._unusable_outputs_total = 0
        self._last_progress = 0.0

    def started(self) -> None:
        with self._lock:
            self._active += 1
            self._requests_total += 1
            self._last_progress = self._clock()

    def progress(self) -> None:
        with self._lock:
            self._last_progress = self._clock()

    def synthesis_failure(self) -> None:
        with self._lock:
            self._synthesis_failures_total += 1

    def unusable_output(self) -> None:
        with self._lock:
            self._unusable_outputs_total += 1

    def finished(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
            if self._active == 0:
                self._last_progress = 0.0

    def snapshot(self) -> tuple[dict[str, Any], int]:
        with self._lock:
            active = self._active
            age = max(0.0, self._clock() - self._last_progress) if active else 0.0
            stalled = bool(active and age >= self.stall_timeout_seconds)
            payload = {
                "status": "degraded" if stalled else "ok",
                "active_requests": active,
                "requests_total": self._requests_total,
                "synthesis_failures_total": self._synthesis_failures_total,
                "unusable_outputs_total": self._unusable_outputs_total,
                "last_progress_age_ms": int(age * 1000.0),
                "stalled": stalled,
            }
        return payload, 503 if stalled else 200


@dataclass(frozen=True)
class SpeechOptions:
    voice: str
    response_format: str
    instructions: Optional[str]
    sampling: Mapping[str, Any]
    language: str


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    voice: str
    response_format: str
    instructions: Optional[str]
    sampling: Mapping[str, Any]
    language: str


class SynthesisState:
    def __init__(
        self,
        response_format: str,
        *,
        request_id: str,
        output_queue_chunks: int,
        deadline: float,
    ) -> None:
        self.response_format = response_format
        self.request_id = request_id
        self.output: queue_module.Queue[Any] = queue_module.Queue(maxsize=output_queue_chunks)
        self.chunks: list[bytes] = []
        self.header_ready = threading.Event()
        self.done = threading.Event()
        self.engine_done = threading.Event()
        self.cancelled = threading.Event()
        self.deadline = deadline
        self.timeout_timer: Optional[threading.Timer] = None
        self.audible = False
        self.received_pcm = False
        self.received_bytes = 0
        self.output_overflow = False
        self.timed_out = False
        self.abort_error = False
        self.error_type: Optional[str] = None
        self.error_message: Optional[str] = None
        self.error_status = 502
        self.worker: Optional[threading.Thread] = None
        self._lifecycle_lock = threading.Lock()
        self._control = threading.Condition()
        self._engine_active = False
        self._pause_requested = False

    def begin_engine(self) -> bool:
        with self._control:
            while self._pause_requested and not self.cancelled.is_set():
                self._control.wait()
            if self.cancelled.is_set():
                return False
            self._engine_active = True
            return True

    def end_engine(self) -> None:
        with self._control:
            self._engine_active = False
            self._control.notify_all()

    def engine_active(self) -> bool:
        with self._control:
            return self._engine_active

    def request_pause(self) -> None:
        with self._control:
            self._pause_requested = True
            self._control.notify_all()

    def request_resume(self) -> None:
        with self._control:
            self._pause_requested = False
            self._control.notify_all()

    def pause_requested(self) -> bool:
        with self._control:
            return self._pause_requested

    def wait_until_resumed_or_cancelled(self) -> bool:
        with self._control:
            while self._pause_requested and not self.cancelled.is_set():
                self._control.wait()
            return not self.cancelled.is_set()

    def wake(self) -> None:
        with self._control:
            self._control.notify_all()


class _CaptureQueue:
    def __init__(
        self,
        state: SynthesisState,
        metrics: RequestMetrics,
        *,
        abort: Callable[[SynthesisState, str, str, int], None],
        max_output_bytes: int,
    ) -> None:
        self.state = state
        self.metrics = metrics
        self.abort = abort
        self.max_output_bytes = max_output_bytes

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:
        del block, timeout
        if self.state.cancelled.is_set():
            return
        pcm = bytes(item)
        if not pcm:
            return
        if self.state.received_bytes + len(pcm) > self.max_output_bytes:
            self.abort(
                self.state,
                "server_error",
                "synthesis output exceeded the configured limit",
                503,
            )
            return
        self.state.received_bytes += len(pcm)
        if self.state.response_format == "wav":
            self.state.chunks.append(pcm)
            self.state.received_pcm = True
            self.metrics.progress()
            return
        if not self.state.audible and _pcm_is_audible(pcm):
            self.state.audible = True
            self.state.header_ready.set()
        while not self.state.cancelled.is_set():
            try:
                self.state.output.put(pcm, timeout=0.1)
                self.state.received_pcm = True
                self.metrics.progress()
                return
            except queue_module.Full:
                if time.monotonic() >= self.state.deadline:
                    self.abort(
                        self.state,
                        "timeout_error",
                        "synthesis timed out while delivering audio",
                        504,
                    )
                    return


class _DiscardQueue:
    """Queue-compatible sink for startup audio that must never reach a client."""

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:
        del item, block, timeout


class QwenHttpServer:
    def __init__(
        self,
        engine: Any,
        *,
        alias: str = "qwen3-tts-native-q8",
        language: str = "auto",
        voice_dir: Path,
        startup_warmup_voice: Optional[str] = None,
        startup_warmup_routes: bool = True,
        stall_timeout_seconds: float = DEFAULT_STALL_TIMEOUT_SECONDS,
        synthesis_timeout_seconds: float = DEFAULT_SYNTHESIS_TIMEOUT_SECONDS,
        shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
        max_active_requests: int = DEFAULT_MAX_ACTIVE_REQUESTS,
        max_queued_requests: int = DEFAULT_MAX_QUEUED_REQUESTS,
        output_queue_chunks: int = DEFAULT_OUTPUT_QUEUE_CHUNKS,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        api_key: Optional[str] = None,
        cors_origins: Optional[Sequence[str]] = None,
        studio_peers: Optional[Sequence[Mapping[str, str]]] = None,
        clock: Callable[[], float] = time.monotonic,
        language_router: Optional[QwenLanguageRouter] = None,
        fragment_lookahead_words: int = 0,
        comma_silence_duration: float = 0.15,
        sentence_silence_duration: float = 0.30,
    ) -> None:
        self.engine = engine
        self.studio_peers = list(studio_peers or [])
        self.alias = str(alias).strip()
        self.language = str(language).strip().lower()
        if not self.alias or not self.language:
            raise ValueError("alias and language must not be empty")
        self.startup_warmup_voice = (
            str(startup_warmup_voice).strip() if startup_warmup_voice is not None else None
        )
        if startup_warmup_voice is not None and not self.startup_warmup_voice:
            raise ValueError("startup_warmup_voice must not be empty")
        self.startup_warmup_routes = bool(startup_warmup_routes)
        if synthesis_timeout_seconds <= 0:
            raise ValueError("synthesis_timeout_seconds must be positive")
        if shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")
        if max_active_requests != 1:
            raise ValueError(
                "max_active_requests must be 1 because the Qwen engine is shared"
            )
        if max_queued_requests < 0:
            raise ValueError("max_queued_requests must not be negative")
        if output_queue_chunks <= 0:
            raise ValueError("output_queue_chunks must be positive")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        if fragment_lookahead_words < 0:
            raise ValueError("fragment_lookahead_words must not be negative")
        if comma_silence_duration < 0 or sentence_silence_duration < 0:
            raise ValueError("punctuation silence durations must not be negative")
        self.registry = VoiceRegistry(voice_dir)
        self.metrics = RequestMetrics(stall_timeout_seconds, clock)
        self.language_router = language_router
        self.synthesis_timeout_seconds = float(synthesis_timeout_seconds)
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self.max_active_requests = int(max_active_requests)
        self.max_queued_requests = int(max_queued_requests)
        self.output_queue_chunks = int(output_queue_chunks)
        self.max_output_bytes = int(max_output_bytes)
        self.fragment_lookahead_words = int(fragment_lookahead_words)
        self.comma_silence_duration = float(comma_silence_duration)
        self.sentence_silence_duration = float(sentence_silence_duration)
        self.api_key = str(api_key) if api_key else None
        configured_origins = DEFAULT_CORS_ORIGINS if cors_origins is None else cors_origins
        self.cors_origins = tuple(str(origin).strip() for origin in configured_origins if str(origin).strip())
        if any(origin == "*" for origin in self.cors_origins):
            raise ValueError(
                "wildcard CORS is not supported; configure explicit --cors-origin values"
            )
        self._operation_lock = threading.RLock()
        self._startup_lock = threading.Lock()
        self._startup_complete = False
        self._states_lock = threading.Lock()
        self._states: set[SynthesisState] = set()
        self._request_slots = threading.BoundedSemaphore(
            self.max_active_requests + self.max_queued_requests
        )
        self._shutting_down = threading.Event()
        self._shutdown_complete = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._sampling_defaults = {
            name: getattr(engine, name)
            for name in (
                "seed",
                "max_new_tokens",
                "temperature",
                "top_k",
                "top_p",
                "repetition_penalty",
                "clone_mode",
                "do_sample",
                "subtalker_do_sample",
                "subtalker_temperature",
                "subtalker_top_k",
                "subtalker_top_p",
            )
            if hasattr(engine, name)
        }

    def fragment_silence_duration(self, text: str) -> float:
        """Return the configured pause after one synthesized text fragment."""
        stripped = text.strip()
        if not stripped:
            return 0.0
        if stripped[-1] in _END_SENTENCE_DELIMITERS:
            return self.sentence_silence_duration
        if stripped[-1] in _MID_SENTENCE_DELIMITERS:
            return self.comma_silence_duration
        return 0.0

    def startup(self) -> None:
        """Prepare configured persistent voices before the app becomes ready."""

        with self._startup_lock:
            if self._startup_complete:
                return
            warmup_targets: list[tuple[str, str]] = []
            if self.language_router is not None:
                started = time.monotonic()
                self.language_router.warmup()
                for base_voice, routes in self.language_router.voice_routes.items():
                    for route_language, route_voice in routes.items():
                        if self.registry.get(route_voice) is None:
                            raise RuntimeError(
                                f"language route {base_voice!r} targets unknown voice "
                                f"{route_voice!r}"
                            )
                        target = (route_voice, route_language)
                        if self.startup_warmup_routes and target not in warmup_targets:
                            warmup_targets.append(target)
                LOGGER.info(
                    "Warmed local language detector in %.1f ms",
                    (time.monotonic() - started) * 1000.0,
                )
            if self.startup_warmup_voice is not None:
                if not any(
                    voice == self.startup_warmup_voice
                    for voice, _language in warmup_targets
                ):
                    warmup_targets.append((self.startup_warmup_voice, self.language))
            for voice, language in warmup_targets:
                self.warmup_voice(voice, language=language)
            self._startup_complete = True

    def is_ready(self) -> bool:
        with self._startup_lock:
            return self._startup_complete and not self._shutting_down.is_set()

    def capabilities(self) -> dict[str, Any]:
        engine_name = str(getattr(self.engine, "engine_name", "qwen"))
        model_id = str(getattr(self.engine, "model_id", self.alias))
        quant = str(getattr(self.engine, "quant", "unknown"))
        binding_version = str(getattr(self.engine, "binding_version", "unknown"))
        native_abi_version = getattr(self.engine, "native_abi_version", None)
        native_version = str(getattr(self.engine, "native_version", "unknown"))
        device = str(getattr(self.engine, "device", "native"))
        return {
            "object": "capabilities",
            "protocol": SERVER_PROTOCOL,
            "protocol_version": SERVER_PROTOCOL_VERSION,
            "server_version": _server_version(),
            "model": {
                "id": self.alias,
                "model_id": model_id,
                "object": "model",
                "owned_by": "local",
            },
            "engine": {
                "name": engine_name,
                "backend": "qwentts.cpp",
                "device": device,
                "cpu_threads": getattr(self.engine, "cpu_threads", None),
                "cpu_only": bool(getattr(self.engine, "cpu_only", False)),
                "clone_mode": str(self._sampling_defaults.get("clone_mode", "auto")),
                "clone_modes": ["auto", "speaker_only"] if getattr(self.engine, "model_type", "base") == "base" else [],
                "model_id": model_id,
                "quant": quant,
                "binding_version": binding_version,
                "native_abi_version": native_abi_version,
                "native_version": native_version,
                "use_fa": bool(getattr(self.engine, "use_fa", False)),
            },
            "audio": {
                **AUDIO_METADATA,
                "response_formats": ["pcm", "wav"],
                "pcm_media_type": "audio/pcm",
                "wav_media_type": "audio/wav",
            },
            "endpoints": {
                "health": "/health",
                "ready": "/ready",
                "models": "/v1/models",
                "voices": "/v1/audio/voices",
                "speech": "/v1/audio/speech",
                "speech_stream": "/v1/audio/speech-stream",
                "capabilities": "/v1/capabilities",
            },
            "stream_controls": {
                "input_events": ["pause", "resume", "cancel", "flush", "segment_start", "skip_segment"],
                "segmented_narration": True,
                "pause_acknowledgement": "native_checkpoint",
            },
            "limits": {
                "max_request_bytes": MAX_REQUEST_BYTES,
                "max_stream_event_bytes": MAX_STREAM_EVENT_BYTES,
                "max_stream_text_bytes": MAX_STREAM_TEXT_BYTES,
                "max_active_requests": self.max_active_requests,
                "max_queued_requests": self.max_queued_requests,
                "output_queue_chunks": self.output_queue_chunks,
                "max_output_bytes": self.max_output_bytes,
                "synthesis_timeout_seconds": self.synthesis_timeout_seconds,
                "fragment_lookahead_words": self.fragment_lookahead_words,
            },
            "authentication": {"required": self.api_key is not None, "scheme": "bearer"},
            "sampling_defaults": dict(self._sampling_defaults),
            "studio_peers": self.studio_peers,
            "features": {
                "model_type": getattr(self.engine, "model_type", "base"),
                "voice_cloning": getattr(self.engine, "model_type", "base") == "base",
                "instruction_control": bool(getattr(self.engine, "instruction_control", False)),
            },
            "playback": {
                "startup_buffer_ms": getattr(self.engine, "startup_buffer_ms", None),
                "trim_silence": getattr(self.engine, "trim_silence", None),
                "onset_silence_recovery": bool(
                    getattr(self.engine, "onset_silence_recovery", False)
                ),
            },
            "ready": self.is_ready(),
        }

    def authorized(self, authorization: Optional[str], query_key: Optional[str] = None) -> bool:
        if self.api_key is None:
            return True
        token = ""
        if isinstance(authorization, str) and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if not token and isinstance(query_key, str):
            token = query_key
        return bool(token) and secrets.compare_digest(token, self.api_key)

    def _abort_state(
        self,
        state: SynthesisState,
        error_type: str,
        message: str,
        status: int,
    ) -> None:
        if state.error_type is None:
            state.error_type = error_type
            state.error_message = message
            state.error_status = int(status)
        state.abort_error = True
        self.cancel(state)

    def _timeout_state(self, state: SynthesisState) -> None:
        with state._lifecycle_lock:
            if state.done.is_set() or state.engine_done.is_set():
                return
            state.timed_out = True
        self._abort_state(state, "timeout_error", "synthesis timed out", 504)

    def warmup_voice(self, name: str, *, language: Optional[str] = None) -> None:
        voice_name = str(name).strip()
        voice_language = (
            self.language
            if language is None
            else str(language).strip().lower()
        )
        if not voice_language:
            raise ValueError("startup warmup language must not be empty")
        if getattr(self.engine, "model_type", "base") == "base" and self.registry.get(voice_name) is None:
            raise RuntimeError(f"startup warmup voice {voice_name!r} is not registered in {self.registry.root}")
        selected_voice = self.synthesis_voice(voice_name, voice_language,
            "A clear, natural speaking voice." if getattr(self.engine, "model_type", "base") == "voice_design" else None)

        started = time.monotonic()
        with self._operation_lock:
            if self._shutting_down.is_set():
                raise RuntimeError("server is shutting down")
            original_queue = self.engine.queue
            self.engine.queue = _DiscardQueue()
            try:
                self.engine.set_voice(selected_voice)
                self.engine.warmup()
            finally:
                self.engine.queue = original_queue
        LOGGER.info(
            "Warmed persistent Qwen voice %r for language %s in %.1f ms",
            voice_name,
            voice_language,
            (time.monotonic() - started) * 1000.0,
        )

    def _parse_voice_upload(
        self, payload: Mapping[str, Any]
    ) -> tuple[str, str, str, tuple[bytes, ...]]:
        name_value = payload.get("name")
        if not isinstance(name_value, str) or not name_value.strip():
            raise ApiError(400, "invalid_request_error", "'name' must be a non-empty string")
        name = name_value.strip()
        if len(name) > 256 or "\x00" in name:
            raise ApiError(400, "invalid_request_error", "'name' is out of domain")
        ref_text_value = payload.get("ref_text", "")
        if ref_text_value is None:
            ref_text_value = ""
        if not isinstance(ref_text_value, str):
            raise ApiError(400, "invalid_request_error", "'ref_text' must be a string")
        ref_text = ref_text_value.strip()
        wav = _decode_base64(payload.get("wav_b64"), "wav_b64")
        spk = _decode_base64(payload.get("spk_b64"), "spk_b64")
        rvq = _decode_base64(payload.get("rvq_b64"), "rvq_b64")
        has_latents = spk is not None and rvq is not None
        if (spk is None) != (rvq is None) or (wav is not None) == has_latents:
            raise ApiError(
                400,
                "invalid_request_error",
                "provide either 'wav_b64' or both 'spk_b64' and 'rvq_b64'",
            )
        if wav is not None:
            return name, ref_text, "wav", (wav,)
        return name, ref_text, "latents", (spk, rvq)  # type: ignore[arg-type]

    def synthesis_voice(self, name: str, language: str, instructions: Optional[str] = None) -> QwenVoice:
        mode = getattr(self.engine, "model_type", "base")
        if mode == "custom_voice":
            return QwenVoice(name=name, speaker=name, language=language, instruct=instructions)
        if mode == "voice_design":
            return QwenVoice(name="design", language=language, instruct=instructions)
        record = self.registry.get(name)
        if record is None:
            raise RuntimeError(f"registered voice disappeared: {name}")
        return self.registry.to_voice(record, language, instructions)

    def register_voice(self, payload: Mapping[str, Any]) -> VoiceRecord:
        if getattr(self.engine, "model_type", "base") != "base":
            raise ApiError(400, "unsupported_operation", "reference cloning requires a Base model")
        name, ref_text, kind, payloads = self._parse_voice_upload(payload)
        record: Optional[VoiceRecord] = None
        try:
            record = self.registry.candidate(name, ref_text, kind, payloads)
            with self._operation_lock:
                if self._shutting_down.is_set():
                    raise ApiError(503, "server_error", "server is shutting down")
                self.engine.set_voice(self.registry.to_voice(record, self.language))
                self.registry.commit(record)
        except ApiError:
            if record is not None:
                self.registry.discard_candidate(record)
            raise
        except BaseException as exc:
            if record is not None:
                self.registry.discard_candidate(record)
            raise ApiError(
                400,
                "invalid_request_error",
                str(exc) or "voice registration failed",
            ) from exc
        assert record is not None
        return record

    def remove_voice(self, name: str) -> bool:
        with self._operation_lock:
            return self.registry.remove(name)

    def parse_speech_options(self, payload: Mapping[str, Any]) -> SpeechOptions:
        mode = getattr(self.engine, "model_type", "base")
        if payload.get("model", self.alias) != self.alias:
            raise ApiError(400, "invalid_request_error", "requested model is not loaded on this server")
        voice = payload.get("voice", "design" if mode == "voice_design" else "")
        if not isinstance(voice, str) or not voice.strip():
            raise ApiError(400, "invalid_request_error", "'voice' must name a registered voice")
        voice = voice.strip()
        if mode == "base" and self.registry.get(voice) is None:
            raise ApiError(400, "invalid_request_error", f"unknown voice: {voice}")
        if mode == "custom_voice" and voice not in getattr(self.engine, "built_in_speakers", ()):
            raise ApiError(400, "invalid_request_error", f"unknown built-in speaker: {voice}")
        response_format = payload.get("response_format", "pcm")
        if not isinstance(response_format, str) or response_format not in {"pcm", "wav"}:
            raise ApiError(400, "invalid_request_error", "response_format must be 'pcm' or 'wav'")
        instructions = payload.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise ApiError(400, "invalid_request_error", "'instructions' must be a string")
        instructions = instructions.strip() if instructions else None
        if instructions and hasattr(self.engine, "instruction_control") and not self.engine.instruction_control:
            raise ApiError(400, "unsupported_operation", "this model does not support instruction control")
        if mode == "voice_design" and not instructions:
            raise ApiError(400, "invalid_request_error", "VoiceDesign requires non-empty instructions")
        language = payload.get("language", self.language)
        if not isinstance(language, str) or not language.strip():
            raise ApiError(
                400,
                "invalid_request_error",
                "'language' must be a non-empty string",
            )
        language = language.strip().lower()

        sampling: dict[str, Any] = {}
        if "clone_mode" in payload:
            if mode != "base":
                raise ApiError(400, "unsupported_operation", "clone_mode requires a Base model")
            clone_mode = payload["clone_mode"]
            if not isinstance(clone_mode, str) or clone_mode not in {"auto", "speaker_only"}:
                raise ApiError(400, "invalid_request_error", "'clone_mode' must be 'auto' or 'speaker_only'")
            sampling["clone_mode"] = clone_mode
        integer_domains = {
            "seed": (-(2**63), 2**63 - 1),
            "max_new_tokens": (1, 2**31 - 1),
            "top_k": (0, 2**31 - 1),
            "subtalker_top_k": (0, 2**31 - 1),
        }
        float_domains = {
            "temperature": (0.0, float("inf"), True),
            "top_p": (0.0, 1.0, False),
            "repetition_penalty": (0.0, float("inf"), False),
            "subtalker_temperature": (0.0, float("inf"), True),
            "subtalker_top_p": (0.0, 1.0, False),
        }
        for field in ("do_sample", "subtalker_do_sample"):
            if field in payload:
                if not isinstance(payload[field], bool):
                    raise ApiError(400, "invalid_request_error", f"'{field}' must be a boolean")
                sampling[field] = payload[field]
        for field, (low, high) in integer_domains.items():
            if field not in payload:
                continue
            value = payload[field]
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ApiError(400, "invalid_request_error", f"'{field}' is out of domain")
            sampling[field] = value
        for field, (low, high, include_low) in float_domains.items():
            if field not in payload:
                continue
            value = payload[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ApiError(400, "invalid_request_error", f"'{field}' is out of domain")
            numeric = float(value)
            lower_ok = numeric >= low if include_low else numeric > low
            if not lower_ok or numeric > high or not np.isfinite(numeric):
                raise ApiError(400, "invalid_request_error", f"'{field}' is out of domain")
            sampling[field] = numeric
        return SpeechOptions(voice, response_format, instructions, sampling, language)

    def parse_speech(self, payload: Mapping[str, Any]) -> SpeechRequest:
        text = payload.get("input")
        if not isinstance(text, str) or not text.strip():
            raise ApiError(400, "invalid_request_error", "'input' must be a non-empty string")
        options = self.parse_speech_options(payload)
        return SpeechRequest(
            text.strip(),
            options.voice,
            options.response_format,
            options.instructions,
            options.sampling,
            options.language,
        )

    def start_synthesis(
        self, request: SpeechRequest, *, request_id: Optional[str] = None
    ) -> SynthesisState:
        if self._shutting_down.is_set():
            raise ApiError(503, "server_error", "server is shutting down")
        if not self._request_slots.acquire(blocking=False):
            raise ApiError(429, "rate_limit_error", "server is busy; try again later")
        state = SynthesisState(
            request.response_format,
            request_id=_new_client_id(request_id),
            output_queue_chunks=self.output_queue_chunks,
            deadline=time.monotonic() + self.synthesis_timeout_seconds,
        )
        with self._states_lock:
            self._states.add(state)
        worker = threading.Thread(
            target=self._synthesis_worker,
            args=(request, state),
            name=f"qwen-http-{uuid.uuid4().hex[:8]}",
            daemon=True,
        )
        state.worker = worker
        try:
            worker.start()
        except BaseException:
            with self._states_lock:
                self._states.discard(state)
            self._request_slots.release()
            raise
        timer = threading.Timer(
            self.synthesis_timeout_seconds,
            self._timeout_state,
            args=(state,),
        )
        timer.daemon = True
        state.timeout_timer = timer
        timer.start()
        return state

    def route_request(
        self,
        request: SpeechRequest,
        *,
        fallback_language: Optional[str] = None,
    ) -> tuple[SpeechRequest, Optional[LanguageDetection]]:
        if self.language_router is None:
            return request, None
        route = self.language_router.resolve(
            request.text,
            base_voice=request.voice,
            language=request.language,
            fallback_language=fallback_language,
        )
        if route.detection is not None:
            LOGGER.info(
                "Language route label=%s score=%s fallback=%s language=%s "
                "voice=%s->%s detection=%.3fms",
                route.detection.label,
                route.detection.confidence,
                route.detection.used_fallback,
                route.language,
                request.voice,
                route.voice,
                route.detection_latency_ms,
            )
        return replace(request, voice=route.voice, language=route.language), route.detection

    def _synthesis_worker(self, request: SpeechRequest, state: SynthesisState) -> None:
        self.metrics.started()
        try:
            request, _detection = self.route_request(request)
            selected_voice = self.synthesis_voice(request.voice, request.language, request.instructions)
            with self._operation_lock:
                if state.cancelled.is_set() or self._shutting_down.is_set():
                    return
                parameters = dict(self._sampling_defaults)
                parameters.update(request.sampling)
                if parameters:
                    self.engine.set_voice_parameters(**parameters)
                self.engine.set_voice(selected_voice)
                if state.cancelled.is_set():
                    return
                original_queue = self.engine.queue
                self.engine.queue = _CaptureQueue(
                    state,
                    self.metrics,
                    abort=self._abort_state,
                    max_output_bytes=self.max_output_bytes,
                )
                try:
                    if not state.begin_engine():
                        return
                    try:
                        ok = bool(self.engine.synthesize(request.text))
                    finally:
                        state.end_engine()
                        resume = getattr(self.engine, "resume", None)
                        if callable(resume):
                            resume()
                finally:
                    self.engine.queue = original_queue
                    with state._lifecycle_lock:
                        state.engine_done.set()
            if state.cancelled.is_set():
                return
            if request.response_format == "wav":
                state.audible = _pcm_is_audible(b"".join(state.chunks))
            if not ok:
                error = getattr(self.engine, "last_error", None)
                message = str(error or "synthesis failed")
                if "completed without producing audio" in message:
                    state.error_type = "output_error"
                    state.error_message = "synthesis produced no audible audio"
                    self.metrics.unusable_output()
                    LOGGER.warning(
                        "Qwen synthesis produced no audible audio for voice %s",
                        request.voice,
                    )
                else:
                    state.error_type = "server_error"
                    state.error_message = message
                    self.metrics.synthesis_failure()
                    LOGGER.error("Qwen synthesis failed for voice %s: %s", request.voice, message)
            elif not state.audible:
                state.error_type = "output_error"
                state.error_message = "synthesis produced no audible audio"
                self.metrics.unusable_output()
                LOGGER.warning(
                    "Qwen synthesis produced no audible audio for voice %s",
                    request.voice,
                )
        except BaseException as exc:
            if not state.cancelled.is_set():
                state.error_type = "server_error"
                state.error_message = str(exc) or exc.__class__.__name__
                self.metrics.synthesis_failure()
                LOGGER.exception("Qwen HTTP synthesis worker failed")
        finally:
            state.end_engine()
            with state._lifecycle_lock:
                state.engine_done.set()
            self.metrics.finished()
            timer = state.timeout_timer
            if timer is not None:
                timer.cancel()
            state.done.set()
            state.header_ready.set()
            if state.response_format != "wav":
                try:
                    state.output.put(_STREAM_END, timeout=0.1)
                except queue_module.Full:
                    # A canceled or silent request may never have a consumer.
                    # Drop buffered PCM so the worker can always publish the
                    # terminal marker without blocking shutdown.
                    while True:
                        try:
                            state.output.get_nowait()
                        except queue_module.Empty:
                            break
                    state.output.put_nowait(_STREAM_END)
            with self._states_lock:
                self._states.discard(state)
            self._request_slots.release()

    def pause(self, state: SynthesisState, timeout: float = PAUSE_ACK_TIMEOUT_SECONDS) -> bool:
        state.request_pause()
        if not state.engine_active():
            return True
        pause = getattr(self.engine, "pause", None)
        if not callable(pause):
            return False
        started = time.monotonic()
        if bool(pause(timeout=timeout)):
            return True
        remaining = max(0.0, timeout - (time.monotonic() - started))
        return state.engine_done.wait(remaining) or not state.engine_active()

    def resume(self, state: SynthesisState) -> None:
        state.request_resume()
        if not state.engine_active():
            return
        resume = getattr(self.engine, "resume", None)
        if callable(resume):
            resume()

    def cancel(self, state: SynthesisState) -> None:
        state.cancelled.set()
        state.wake()
        if state.response_format != "wav":
            # Discard buffered PCM and wake a WebSocket/HTTP consumer that is
            # blocked in Queue.get(). The capture queue observes cancelled and
            # will not publish new chunks after this point.
            while True:
                try:
                    state.output.get_nowait()
                except queue_module.Empty:
                    try:
                        state.output.put_nowait(_STREAM_END)
                    except queue_module.Full:
                        # One producer put may have raced with cancellation.
                        # Drain it and retry; subsequent puts see cancelled.
                        continue
                    break
        if state.engine_active():
            self.engine.stop()

    def iter_pcm(self, state: SynthesisState) -> Iterator[bytes]:
        try:
            while True:
                chunk = state.output.get()
                if chunk is _STREAM_END:
                    break
                if state.cancelled.is_set():
                    break
                yield chunk
        finally:
            if not state.done.is_set():
                self.cancel(state)

    def shutdown(self) -> None:
        if self._shutdown_complete.is_set():
            return
        self._shutting_down.set()
        with self._states_lock:
            states = list(self._states)
        for state in states:
            self.cancel(state)
        shutdown_deadline = time.monotonic() + self.shutdown_timeout_seconds
        for state in states:
            worker = state.worker
            if worker is not None:
                worker.join(timeout=max(0.0, shutdown_deadline - time.monotonic()))
        alive = [
            worker
            for state in states
            if (worker := state.worker) is not None and worker.is_alive()
        ]
        if alive:
            LOGGER.error(
                "Refusing to close Qwen engine while %d synthesis worker(s) remain alive",
                len(alive),
            )
            return
        with self._shutdown_lock:
            if self._shutdown_complete.is_set():
                return
            self.engine.shutdown()
            self._shutdown_complete.set()


async def _json_object(request: Any) -> Mapping[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                raise ApiError(413, "invalid_request_error", "request body is too large")
        except ValueError as exc:
            raise ApiError(400, "invalid_request_error", "invalid Content-Length") from exc
    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        raise ApiError(413, "invalid_request_error", "request body is too large")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError(400, "invalid_request_error", "request body is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ApiError(400, "invalid_request_error", "request body must be a JSON object")
    return payload


async def _stream_token_source(
    items: asyncio.Queue[Any], ended: Optional[asyncio.Event] = None,
    boundaries: Optional[list[_SegmentBoundary]] = None,
) -> AsyncIterator[str]:
    while True:
        item = await items.get()
        if item is _TEXT_END:
            if ended is not None:
                ended.set()
            return
        if item is _TEXT_FLUSH:
            return
        if isinstance(item, _SegmentBoundary):
            if boundaries is not None:
                boundaries.append(item)
            return
        yield str(item)


async def _stream_sentence_fragments(items: asyncio.Queue[Any], **options: Any) -> AsyncIterator[Any]:
    """Flush an explicitly finished speech segment without closing the session.

    A model tool call ends its preceding narration. Waiting for the next model
    round as sentence lookahead would hide that narration behind tool execution.
    The ordinary splitter still handles all text inside each segment unchanged.
    """
    ended = asyncio.Event()
    while not ended.is_set():
        boundaries: list[_SegmentBoundary] = []
        async for fragment in generate_sentences_async(_stream_token_source(items, ended, boundaries), **options):
            yield fragment
        for boundary in boundaries:
            yield boundary


async def _websocket_json(websocket: Any) -> Mapping[str, Any]:
    raw = await websocket.receive_text()
    if len(raw.encode("utf-8")) > MAX_STREAM_EVENT_BYTES:
        raise ApiError(413, "invalid_request_error", "stream event is too large")
    try:
        event = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApiError(400, "invalid_request_error", "stream event is not valid JSON") from exc
    if not isinstance(event, dict):
        raise ApiError(400, "invalid_request_error", "stream event must be a JSON object")
    return event


async def _wait_until_stream_resumed(
    resumed: asyncio.Event,
    cancelled: asyncio.Event,
    state: Optional[SynthesisState] = None,
) -> bool:
    if resumed.is_set():
        return not cancelled.is_set() and not (
            state is not None and state.cancelled.is_set()
        )
    resume_task = asyncio.create_task(resumed.wait())
    cancel_task = asyncio.create_task(cancelled.wait())
    tasks: list[asyncio.Task[Any]] = [resume_task, cancel_task]
    state_task: Optional[asyncio.Task[Any]] = None
    if state is not None:
        state_task = asyncio.create_task(
            asyncio.to_thread(state.wait_until_resumed_or_cancelled)
        )
        tasks.append(state_task)
    _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    for task in pending:
        with suppress(asyncio.CancelledError):
            await task
    if cancelled.is_set() or (state is not None and state.cancelled.is_set()):
        return False
    if resumed.is_set():
        return True
    # The state condition is released before the receiver emits the resumed
    # acknowledgement and opens the PCM gate. Wait for that ordered commit.
    return await _wait_until_stream_resumed(resumed, cancelled)


async def _send_websocket_json(
    websocket: Any,
    event: Mapping[str, Any],
    send_lock: Optional[asyncio.Lock] = None,
) -> None:
    if send_lock is None:
        await websocket.send_json(dict(event))
        return
    async with send_lock:
        await websocket.send_json(dict(event))


async def _send_stream_pcm(
    websocket: Any,
    state: SynthesisState,
    *,
    synthesis_started: float,
    first_fragment_timings: Optional[Mapping[str, Any]],
    session_id: str,
    cancelled: asyncio.Event,
    resumed: Optional[asyncio.Event] = None,
    send_lock: Optional[asyncio.Lock] = None,
) -> None:
    first_chunk = True

    async def send_chunk(chunk: Any) -> bool:
        nonlocal first_chunk
        if first_chunk and first_fragment_timings is not None:
            timings = dict(first_fragment_timings)
            timings["synthesis_to_first_pcm_ms"] = round(
                (time.monotonic() - synthesis_started) * 1000.0,
                3,
            )
            await websocket.send_json(
                {
                    "type": "first_pcm_ready",
                    "session_id": session_id,
                    "request_id": state.request_id,
                    "audio": dict(AUDIO_METADATA),
                    "timings": timings,
                }
            )
        if cancelled.is_set() or state.cancelled.is_set():
            return False
        await websocket.send_bytes(bytes(chunk))
        first_chunk = False
        return True

    while not cancelled.is_set() and not state.cancelled.is_set():
        try:
            chunk = await asyncio.to_thread(state.output.get, True, 0.1)
        except queue_module.Empty:
            if cancelled.is_set() or state.cancelled.is_set():
                break
            if time.monotonic() >= state.deadline:
                raise ApiError(504, "timeout_error", "synthesis timed out")
            continue
        if chunk is _STREAM_END:
            break
        if cancelled.is_set() or state.cancelled.is_set():
            break
        sent = False
        while not cancelled.is_set() and not state.cancelled.is_set():
            if resumed is not None and not await _wait_until_stream_resumed(
                resumed, cancelled, state
            ):
                break
            if send_lock is None:
                if resumed is not None and not resumed.is_set():
                    continue
                sent = await send_chunk(chunk)
            else:
                async with send_lock:
                    if resumed is not None and not resumed.is_set():
                        continue
                    sent = await send_chunk(chunk)
            if sent or cancelled.is_set() or state.cancelled.is_set():
                break
        if not sent:
            break
    if state.error_type is not None and (
        not state.cancelled.is_set() or state.abort_error
    ):
        raise ApiError(
            state.error_status,
            state.error_type,
            state.error_message or "synthesis failed",
        )


async def _send_stream_silence(
    websocket: Any,
    duration: float,
    *,
    cancelled: asyncio.Event,
    resumed: asyncio.Event,
    send_lock: asyncio.Lock,
) -> None:
    silent_frames = int(SAMPLE_RATE * duration)
    if silent_frames <= 0:
        return
    silence = b"\0" * (
        silent_frames
        * int(AUDIO_METADATA["channels"])
        * int(AUDIO_METADATA["bits_per_sample"])
        // 8
    )
    while not cancelled.is_set():
        if not await _wait_until_stream_resumed(resumed, cancelled):
            return
        async with send_lock:
            if not resumed.is_set():
                continue
            if cancelled.is_set():
                return
            await websocket.send_bytes(silence)
            return


def create_app(server: QwenHttpServer) -> Any:
    if FastAPI is None:  # pragma: no cover - guarded by qwen-server extra
        raise ImportError(
            "The Qwen HTTP server requires FastAPI and Uvicorn. Install "
            "with: pip install \"realtimetts[qwen-server]\""
        )

    @asynccontextmanager
    async def lifespan(_app: Any):
        try:
            await asyncio.to_thread(server.startup)
            yield
        finally:
            await asyncio.to_thread(server.shutdown)

    app = FastAPI(lifespan=lifespan)
    app.state.qwen_server = server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(server.cors_origins),
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID", "X-Session-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[..., Any]) -> Any:
        request_id = _new_client_id(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        if (
            server.api_key is not None
            and request.method != "OPTIONS"
            and request.url.path not in {"/health", "/", "/studio", "/studio/studio.css", "/studio/studio.js"}
            and not server.authorized(request.headers.get("authorization"))
        ):
            response = JSONResponse(
                status_code=401,
                content=_error_payload("authentication_error", "missing or invalid API key"),
                headers={"WWW-Authenticate": "Bearer"},
            )
        else:
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    def api_error(exc: ApiError) -> Any:
        return JSONResponse(
            status_code=exc.status,
            content=_error_payload(exc.error_type, exc.message),
        )

    # Only the credential-free UI assets are public. Every inference and voice
    # operation retains the existing bearer authentication.
    studio_root = Path(__file__).with_name("studio")

    @app.get("/", include_in_schema=False)
    @app.get("/studio", include_in_schema=False)
    def studio() -> Any:
        return FileResponse(studio_root / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/studio/{asset}", include_in_schema=False)
    def studio_asset(asset: str) -> Any:
        if asset not in {"studio.css", "studio.js"}:
            return Response(status_code=404)
        return FileResponse(studio_root / asset, headers={"Cache-Control": "no-cache"})

    @app.get("/health")
    def health() -> Any:
        payload, status = server.metrics.snapshot()
        return JSONResponse(status_code=status, content=payload)

    @app.get("/ready")
    def ready() -> Any:
        if server.is_ready():
            return {"status": "ready"}
        return JSONResponse(status_code=503, content={"status": "not_ready"})

    @app.get("/v1/capabilities")
    def capabilities() -> Any:
        return server.capabilities()

    @app.post("/v1/text/languages")
    async def text_languages(request: Request) -> Any:
        """Reuse the warmed CPU detector without taking the synthesis/engine lock."""
        try:
            payload = await _json_object(request)
            texts = payload.get("texts")
            if (not isinstance(texts, list) or not 1 <= len(texts) <= 128
                    or any(not isinstance(t, str) or not t.strip() for t in texts)
                    or sum(len(t) for t in texts) > 200_000):
                raise ApiError(400, "invalid_request_error", "texts must contain 1-128 non-empty sentences, at most 200000 characters")
            if server.language_router is None:
                raise ApiError(503, "server_error", "language detector is not configured")
            from dataclasses import asdict
            from .language_router import normalize_qwen_language, SUPPORTED_QWEN_LANGUAGES
            fallback = normalize_qwen_language(payload.get("fallback_language", "english"))
            def classify():
                router = server.language_router
                values = []
                for text in texts:
                    detector = router.detector
                    value = (detector.detect(text, fallback_language=fallback)
                             if isinstance(detector, FastTextLanguageDetector)
                             else detector.detect(text))
                    values.append(asdict(router._detector_result(value)))
                return values
            return {"languages": await asyncio.to_thread(classify),
                    "supported_languages": sorted(SUPPORTED_QWEN_LANGUAGES)}
        except ApiError as exc:
            return api_error(exc)
        except (TypeError, ValueError) as exc:
            return api_error(ApiError(400, "invalid_request_error", str(exc)))

    @app.get("/v1/models")
    def models() -> Any:
        return {
            "object": "list",
            "data": [{"id": server.alias, "object": "model", "owned_by": "local"}],
        }

    @app.get("/v1/audio/voices")
    def voices() -> Any:
        mode = getattr(server.engine, "model_type", "base")
        if mode == "custom_voice":
            return {"voices": [{"name": name, "kind": "built_in"} for name in server.engine.built_in_speakers]}
        if mode == "voice_design":
            return {"voices": [{"name": "design", "kind": "voice_design"}]}
        return {
            "voices": [
                {"name": record.name, "kind": "registered"}
                for record in server.registry.records()
            ]
        }

    @app.post("/v1/audio/voices")
    async def register_voice(request: Request) -> Any:
        try:
            payload = await _json_object(request)
            record = await asyncio.to_thread(server.register_voice, payload)
            return {"name": record.name, "status": "registered"}
        except ApiError as exc:
            return api_error(exc)

    @app.delete("/v1/audio/voices/{name:path}")
    async def delete_voice(name: str) -> Any:
        removed = await asyncio.to_thread(server.remove_voice, name)
        if not removed:
            return api_error(ApiError(404, "not_found_error", "no registered voice with this name"))
        return {"status": "deleted"}

    @app.post("/v1/audio/speech")
    async def speech(http_request: Request) -> Any:
        try:
            payload = await _json_object(http_request)
            speech_request = server.parse_speech(payload)
            state = server.start_synthesis(
                speech_request,
                request_id=getattr(http_request.state, "request_id", None),
            )
        except ApiError as exc:
            return api_error(exc)

        wait_event = state.done if speech_request.response_format == "wav" else state.header_ready
        while not wait_event.is_set():
            if await http_request.is_disconnected():
                server.cancel(state)
                return Response(status_code=499)
            # Wait in a worker so an audio callback wakes the route immediately;
            # fixed async polling would add up to one whole poll interval to TTFA.
            remaining = state.deadline - time.monotonic()
            if remaining <= 0.0:
                server._timeout_state(state)
                return api_error(ApiError(504, "timeout_error", "synthesis timed out"))
            await asyncio.to_thread(wait_event.wait, min(0.05, remaining))

        if state.error_type is not None:
            return api_error(
                ApiError(
                    state.error_status,
                    state.error_type,
                    state.error_message or "synthesis failed",
                )
            )
        if not state.audible:
            return api_error(
                ApiError(502, "output_error", "synthesis produced no audible audio")
            )
        if speech_request.response_format == "wav":
            return Response(
                content=_pcm_to_wav(b"".join(state.chunks)),
                media_type="audio/wav",
                headers=_audio_response_headers(),
            )
        return StreamingResponse(
            server.iter_pcm(state),
            media_type="audio/pcm",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                **_audio_response_headers(),
            },
        )

    @app.websocket("/v1/audio/speech-stream")
    async def speech_stream(websocket: WebSocket) -> None:
        # Browser WebSocket cannot set Authorization. A non-echoed subprotocol
        # carries its bearer token without putting credentials in request URLs.
        offered = websocket.scope.get("subprotocols", [])
        browser_key = None
        for protocol in offered:
            if protocol.startswith("bearer."):
                try:
                    encoded = protocol[7:]
                    browser_key = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode("utf-8")
                except (ValueError, UnicodeDecodeError, binascii.Error):
                    browser_key = None
                break
        if server.api_key is not None and not server.authorized(
            websocket.headers.get("authorization"),
            browser_key or websocket.query_params.get("api_key"),
        ):
            await websocket.close(code=1008, reason="missing or invalid API key")
            return
        await websocket.accept(subprotocol="qwen-studio" if "qwen-studio" in offered else None)
        session_id = _new_client_id(websocket.headers.get("x-session-id"))
        items: asyncio.Queue[Any] = asyncio.Queue()
        cancelled = asyncio.Event()
        stream_resumed = asyncio.Event()
        stream_resumed.set()
        send_lock = asyncio.Lock()
        input_ended = asyncio.Event()
        config_ready: asyncio.Future[SpeechOptions] = (
            asyncio.get_running_loop().create_future()
        )
        current_state: list[Optional[SynthesisState]] = [None]
        cancelled_states: set[SynthesisState] = set()
        input_segment: Optional[int] = None
        last_segment_id = 0
        active_segment: Optional[int] = None
        segment_cancelled = asyncio.Event()
        skipped_segments: set[int] = set()
        known_segments: set[int] = set()
        completed_segments: set[int] = set()
        cancel_requested = asyncio.Event()
        receiver_error: list[Optional[BaseException]] = [None]
        last_request_id: list[Optional[str]] = [None]
        first_text_received_at: list[Optional[float]] = [None]
        total_text_bytes = 0
        received_text_parts: list[str] = []
        text_updated = asyncio.Event()
        stream_ended = asyncio.Event()
        text_version = 0
        locked_language: Optional[str] = None

        def cancel_current() -> None:
            segment_cancelled.set()
            state = current_state[0]
            if state is None or state in cancelled_states:
                return
            cancelled_states.add(state)
            server.cancel(state)

        async def finish_input() -> None:
            nonlocal input_segment
            if input_ended.is_set():
                return
            input_ended.set()
            stream_ended.set()
            text_updated.set()
            if input_segment is not None:
                await items.put(_SegmentBoundary(input_segment, False))
                input_segment = None
            await items.put(_TEXT_END)

        async def receive_text() -> None:
            nonlocal session_id, text_version, total_text_bytes, input_segment, last_segment_id
            try:
                first = await _websocket_json(websocket)
                if first.get("type") != "config":
                    raise ApiError(
                        400,
                        "invalid_request_error",
                        "first stream event must be a config event",
                    )
                session_id = _new_client_id(first.get("session_id", session_id))
                options = server.parse_speech_options(first)
                if options.response_format != "pcm":
                    raise ApiError(
                        400,
                        "invalid_request_error",
                        "streaming response_format must be 'pcm'",
                    )
                config_ready.set_result(options)
                while True:
                    event = await _websocket_json(websocket)
                    event_type = event.get("type")
                    if event_type == "text":
                        if input_ended.is_set():
                            raise ApiError(
                                400,
                                "invalid_request_error",
                                "text events must precede the end event",
                            )
                        text = event.get("text")
                        if not isinstance(text, str):
                            raise ApiError(
                                400,
                                "invalid_request_error",
                                "text event requires a string",
                            )
                        encoded_size = len(text.encode("utf-8"))
                        total_text_bytes += encoded_size
                        if total_text_bytes > MAX_STREAM_TEXT_BYTES:
                            raise ApiError(
                                413,
                                "invalid_request_error",
                                "stream text is too large",
                            )
                        if text:
                            if first_text_received_at[0] is None:
                                first_text_received_at[0] = time.monotonic()
                            received_text_parts.append(text)
                            text_version += 1
                            text_updated.set()
                            await items.put(text)
                    elif event_type == "segment_start":
                        segment_id = event.get("segment_id")
                        if (input_ended.is_set() or input_segment is not None
                                or type(segment_id) is not int or segment_id <= last_segment_id):
                            raise ApiError(400, "invalid_request_error", "segment_start requires an increasing positive integer ID after the previous segment flush")
                        last_segment_id = segment_id
                        input_segment = segment_id
                        known_segments.add(segment_id)
                        await items.put(_SegmentBoundary(segment_id, True))
                    elif event_type == "skip_segment":
                        segment_id = event.get("segment_id")
                        if type(segment_id) is not int or segment_id not in known_segments:
                            raise ApiError(400, "invalid_request_error", "skip_segment requires a known positive integer segment_id")
                        if segment_id not in completed_segments:
                            skipped_segments.add(segment_id)
                            if active_segment == segment_id:
                                cancel_current()
                    elif event_type == "flush":
                        if input_ended.is_set():
                            raise ApiError(400, "invalid_request_error", "flush events must precede the end event")
                        if "segment_id" in event:
                            segment_id = event["segment_id"]
                            if type(segment_id) is not int or input_segment != segment_id:
                                raise ApiError(400, "invalid_request_error", "flush segment_id must match the open segment")
                            await items.put(_SegmentBoundary(segment_id, False))
                            input_segment = None
                        else:
                            await items.put(_TEXT_FLUSH)
                    elif event_type == "end":
                        if input_ended.is_set():
                            raise ApiError(
                                400,
                                "invalid_request_error",
                                "stream already ended",
                            )
                        await finish_input()
                        # End closes the text channel, not the control channel.
                        # Continue receiving so playback can still be cancelled.
                    elif event_type == "pause":
                        state = current_state[0]
                        if state is not None:
                            state.request_pause()
                        stream_resumed.clear()
                        if state is not None:
                            paused = await asyncio.to_thread(
                                server.pause,
                                state,
                                PAUSE_ACK_TIMEOUT_SECONDS,
                            )
                            if not paused:
                                raise ApiError(
                                    503,
                                    "server_error",
                                    "native synthesis did not acknowledge pause",
                                )
                        await _send_websocket_json(
                            websocket,
                            {
                                "type": "paused",
                                "session_id": session_id,
                                "request_id": (
                                    state.request_id if state is not None else last_request_id[0]
                                ),
                            },
                            send_lock,
                        )
                    elif event_type == "resume":
                        state = current_state[0]
                        if state is not None:
                            await asyncio.to_thread(server.resume, state)
                        async with send_lock:
                            await websocket.send_json(
                                {
                                    "type": "resumed",
                                    "session_id": session_id,
                                    "request_id": (
                                        state.request_id if state is not None else last_request_id[0]
                                    ),
                                }
                            )
                            stream_resumed.set()
                    elif event_type == "cancel":
                        cancel_requested.set()
                        cancelled.set()
                        stream_resumed.set()
                        await finish_input()
                        cancel_current()
                        return
                    else:
                        raise ApiError(
                            400,
                            "invalid_request_error",
                            f"unknown stream event type: {event_type!r}",
                        )
            except asyncio.CancelledError:
                raise
            except WebSocketDisconnect as exc:
                cancelled.set()
                stream_resumed.set()
                receiver_error[0] = exc
                await finish_input()
                cancel_current()
                if not config_ready.done():
                    config_ready.set_exception(exc)
            except BaseException as exc:
                cancelled.set()
                stream_resumed.set()
                receiver_error[0] = exc
                await finish_input()
                cancel_current()
                if not config_ready.done():
                    config_ready.set_exception(exc)


        async def route_first_fragment(request: SpeechRequest) -> SpeechRequest:
            nonlocal locked_language
            started = time.monotonic()
            hard_deadline = started + 0.120
            base_word_count = len(request.text.split())
            last_word_count = -1
            routed = request
            detection: Optional[LanguageDetection] = None
            candidate_language: Optional[str] = None
            decision_deadline = started
            reason = "fallback"
            word_count = base_word_count

            while True:
                probe_text = "".join(received_text_parts).strip() or request.text
                word_count = len(probe_text.split())
                if (
                    last_word_count < 0
                    or word_count > last_word_count
                    or stream_ended.is_set()
                ):
                    probe_request = replace(request, text=probe_text)
                    routed_probe, detection = server.route_request(probe_request)
                    routed = replace(routed_probe, text=request.text)
                    last_word_count = word_count
                    if detection is None or not detection.used_fallback:
                        locked_language = routed.language
                        reason = "confirmed"
                        break
                    wait_ms = language_lookahead_wait_ms(detection)
                    if wait_ms is None:
                        locked_language = routed.language
                        reason = "fallback"
                        break
                    candidate_language = detection.candidate_language
                    decision_deadline = min(
                        hard_deadline, started + wait_ms / 1000.0
                    )

                additional_words = max(0, word_count - base_word_count)
                if stream_ended.is_set():
                    locked_language = candidate_language
                    reason = "stream_end"
                    break
                if additional_words >= LANGUAGE_LOOKAHEAD_MAX_ADDITIONAL_WORDS:
                    locked_language = candidate_language
                    reason = "word_limit"
                    break
                remaining = decision_deadline - time.monotonic()
                if remaining <= 0.0:
                    locked_language = candidate_language
                    reason = "timeout"
                    break

                observed_version = text_version
                text_updated.clear()
                if text_version != observed_version:
                    continue
                try:
                    await asyncio.wait_for(text_updated.wait(), timeout=remaining)
                except TimeoutError:
                    pass

            if locked_language is None:
                locked_language = routed.language
            locked_request = replace(request, language=locked_language)
            locked_request, _ = server.route_request(locked_request)
            LOGGER.info(
                "Language lock language=%s reason=%s wait=%.1fms additional_words=%d",
                locked_language,
                reason,
                (time.monotonic() - started) * 1000.0,
                max(0, word_count - base_word_count),
            )
            return locked_request
        receiver = asyncio.create_task(receive_text())
        fragment_index = 0
        try:
            options = await config_ready
            async for fragment in _stream_sentence_fragments(
                items,
                tokenizer="rule-based",
                language=_QWEN_TO_TOKENIZER_LANGUAGE.get(options.language, "en"),
                minimum_sentence_length=10,
                minimum_first_fragment_length=10,
                quick_yield_single_sentence_fragment=True,
                quick_yield_for_all_sentences=True,
                quick_yield_every_fragment=False,
                sentence_fragment_delimiters=".?!;:,\n…)]}。-—",
                force_first_fragment_after_words=_NO_FORCED_FIRST_FRAGMENT,
                fragment_lookahead_words=server.fragment_lookahead_words,
            ):
                if cancelled.is_set():
                    break
                if receiver_error[0] is not None:
                    raise receiver_error[0]
                if isinstance(fragment, _SegmentBoundary):
                    if fragment.started:
                        active_segment = fragment.segment_id
                        segment_cancelled.clear()
                        if active_segment in skipped_segments:
                            segment_cancelled.set()
                        await _send_websocket_json(websocket, {
                            "type": "segment_started", "segment_id": active_segment,
                        }, send_lock)
                    else:
                        completed_segments.add(fragment.segment_id)
                        await _send_websocket_json(websocket, {
                            "type": "segment_done", "segment_id": fragment.segment_id,
                            "skipped": fragment.segment_id in skipped_segments,
                        }, send_lock)
                        active_segment = None
                    continue
                if active_segment in skipped_segments:
                    continue
                text = fragment.strip()
                # Splitter tails may contain only emoji/punctuation. They have no
                # linguistic content to synthesize; Unicode letters and numbers
                # from every writing system remain eligible, including digits.
                if not text or not any(character.isalnum() for character in text):
                    continue
                fragment_index += 1
                fragment_ready_at = time.monotonic()
                first_fragment_timings: Optional[dict[str, Any]] = None
                fragment_request_id = uuid.uuid4().hex
                last_request_id[0] = fragment_request_id
                if fragment_index == 1 and first_text_received_at[0] is not None:
                    first_fragment_timings = {
                        "first_fragment_chars": len(text),
                        "first_fragment_words": len(text.split()),
                        "first_fragment_boundary": text[-1],
                        "first_text_to_fragment_ms": round(
                            (fragment_ready_at - first_text_received_at[0]) * 1000.0,
                            3,
                        ),
                    }
                    await _send_websocket_json(
                        websocket,
                        {
                            "type": "fragment_ready",
                            "session_id": session_id,
                            "request_id": fragment_request_id,
                            "audio": dict(AUDIO_METADATA),
                            "timings": first_fragment_timings,
                        },
                        send_lock,
                    )
                request = SpeechRequest(
                    text,
                    options.voice,
                    "pcm",
                    options.instructions,
                    options.sampling,
                    options.language,
                )
                if fragment_index == 1:
                    request = await route_first_fragment(request)
                else:
                    request = replace(
                        request, language=locked_language or options.language
                    )
                # A cancel can arrive while fragment metadata or language
                # routing awaits. Never start a worker after that cancel won.
                if not await _wait_until_stream_resumed(
                    stream_resumed, cancelled
                ):
                    break
                if cancelled.is_set():
                    break
                if active_segment in skipped_segments:
                    continue
                synthesis_started = time.monotonic()
                state = server.start_synthesis(request, request_id=fragment_request_id)
                current_state[0] = state
                if cancelled.is_set():
                    server.cancel(state)
                await _send_stream_pcm(
                    websocket,
                    state,
                    synthesis_started=synthesis_started,
                    first_fragment_timings=first_fragment_timings,
                    session_id=session_id,
                    cancelled=cancelled,
                    resumed=stream_resumed,
                    send_lock=send_lock,
                )
                if active_segment is not None and state.cancelled.is_set():
                    # Do not acknowledge completion or start a successor while the
                    # cancelled native worker can still own the engine/queue.
                    if not await asyncio.to_thread(state.done.wait, server.synthesis_timeout_seconds):
                        raise ApiError(503, "server_error", "cancelled segment synthesis did not finish")
                await _send_stream_silence(
                    websocket,
                    server.fragment_silence_duration(text),
                    cancelled=segment_cancelled if active_segment is not None else cancelled,
                    resumed=stream_resumed,
                    send_lock=send_lock,
                )
                current_state[0] = None
            # Terminal commit point: after all synthesis/PCM has completed,
            # stop accepting controls and emit done. A cancel processed before
            # this point wins and emits cancelled instead.
            if not receiver.done():
                receiver.cancel()
            with suppress(asyncio.CancelledError):
                await receiver
            error = receiver_error[0]
            if error is not None and not isinstance(error, WebSocketDisconnect):
                raise error
            if cancel_requested.is_set() and error is None:
                await _send_websocket_json(
                    websocket,
                    {
                        "type": "cancelled",
                        "session_id": session_id,
                        "request_id": last_request_id[0],
                    },
                    send_lock,
                )
            elif not cancelled.is_set() and error is None:
                await _send_websocket_json(
                    websocket,
                    {
                        "type": "done",
                        "session_id": session_id,
                        "request_id": last_request_id[0],
                    },
                    send_lock,
                )
        except WebSocketDisconnect:
            cancelled.set()
            stream_resumed.set()
            cancel_current()
        except BaseException as exc:
            cancelled.set()
            stream_resumed.set()
            cancel_current()
            if isinstance(exc, ApiError):
                error_type = exc.error_type
                message = exc.message
            else:
                error_type = "server_error"
                message = str(exc) or exc.__class__.__name__
                LOGGER.exception("Qwen WebSocket synthesis failed")
            with suppress(Exception):
                await _send_websocket_json(
                    websocket,
                    {
                        "type": "error",
                        "session_id": session_id,
                        "request_id": (
                            current_state[0].request_id
                            if current_state[0] is not None
                            else last_request_id[0]
                        ),
                        "message": message,
                        "error": {"type": error_type, "message": message},
                    },
                    send_lock,
                )
        finally:
            stream_resumed.set()
            if not receiver.done():
                receiver.cancel()
            with suppress(asyncio.CancelledError, WebSocketDisconnect):
                await receiver
            with suppress(Exception):
                await websocket.close()

    return app


def _default_voice_dir() -> Path:
    root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "realtimetts" / "qwen-server" / "voices"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RealtimeTTS native Qwen3-TTS HTTP server")
    parser.add_argument("--model", help="Direct talker GGUF path")
    parser.add_argument("--codec", help="Direct codec GGUF path")
    parser.add_argument("--model-id", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument("--quant", default=DEFAULT_QUANT)
    parser.add_argument("--alias", default="qwen3-tts-native-q8")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--onset-silence-profile",
        default="off",
        help=(
            "Checkpoint-specific x-vector onset suppression profile "
            "(default: off; ICL routes remain unchanged)"
        ),
    )
    parser.add_argument(
        "--onset-silence-recovery",
        action="store_true",
        help=(
            "CPU only: retry once if the first 240 ms of native audio stays quiet; "
            "requires the Q8 v1 onset profile, silence trimming, and a recovery-capable native wheel"
        ),
    )
    parser.add_argument(
        "--fragment-lookahead-words",
        type=int,
        default=0,
        help=(
            "Complete words to buffer before choosing a quick-yield boundary; "
            "sentence endings are preferred over commas"
        ),
    )
    parser.add_argument(
        "--allow-lan",
        action="store_true",
        help="Allow a non-loopback bind (requires --api-key)",
    )
    parser.add_argument(
        "--api-key",
        help=(
            "Bearer token required for every non-health endpoint; mandatory for "
            "non-loopback hosts"
        ),
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Read the bearer token from a file (preferred for service deployments)",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=None,
        help="Allowed browser origin (repeat for multiple origins); defaults to localhost origins",
    )
    parser.add_argument("--lang", default="Auto")
    parser.add_argument("--studio-peer", action="append", default=[], metavar="LABEL=URL",
                        help="Already-loaded model server offered in Studio (repeatable; explicit CORS required)")
    parser.add_argument(
        "--language-id-model",
        type=Path,
        help=(
            "Local fastText language-ID model used when language is Auto; "
            "lid.176.bin is recommended, while compressed lid.176.ftz is supported"
        ),
    )
    parser.add_argument(
        "--voice-language-route",
        action="append",
        default=[],
        metavar="BASE:LANG=VOICE",
        help="Select a registered reference voice for a base voice and language",
    )
    parser.add_argument("--voice-dir", type=Path, default=_default_voice_dir())
    parser.add_argument(
        "--startup-warmup-voice",
        help="Registered persistent voice to prepare and warm before the server becomes ready",
    )
    parser.add_argument(
        "--startup-warmup-routes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prepare and warm every configured --voice-language-route target before "
            "the server becomes ready (default: enabled)"
        ),
    )
    parser.add_argument(
        "--startup-warmup-text",
        default="Warm up the speech engine.",
        help="Hidden synthesis text used by --startup-warmup-voice",
    )
    parser.add_argument(
        "--startup-warmup-tokens",
        type=int,
        default=32,
        help="Maximum hidden synthesis tokens used by --startup-warmup-voice",
    )
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--library-path", type=Path)
    parser.add_argument(
        "--device", choices=("native", "cpu"), default="native",
        help="Use the default native runtime or the CPU-only Qwen engine",
    )
    parser.add_argument(
        "--cpu-threads", type=int, default=None,
        help="CPU worker count (requires --device cpu; default: up to 8)",
    )
    parser.add_argument(
        "--clone-mode", choices=("auto", "speaker_only"), default=None,
        help="Cloning default: speaker_only for CPU; auto follows reference transcripts for native",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-fa", action="store_true", help="Disable native Flash Attention")
    parser.add_argument(
        "--clamp-fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Clamp FP16 hidden states to prevent overflow artifacts on pre-Ampere "
            "NVIDIA GPUs (default: enabled for native, disabled for CPU)"
        ),
    )
    parser.add_argument("--codec-chunk-sec", type=float, default=24.0)
    parser.add_argument(
        "--trim-silence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim leading silence while preserving onset pre-roll (default: enabled)",
    )
    parser.add_argument("--silence-threshold", type=float, default=0.005)
    parser.add_argument("--trim-pre-roll-ms", type=float, default=15.0)
    parser.add_argument("--trim-fade-in-ms", type=float, default=15.0)
    parser.add_argument(
        "--fragment-fade-out-after-ms",
        type=float,
        default=1000.0,
        help="Stream this much fragment audio before retaining the final fade-out tail",
    )
    parser.add_argument(
        "--comma-silence-duration",
        type=float,
        default=0.15,
        help="Silence in seconds after a comma or other mid-sentence delimiter",
    )
    parser.add_argument(
        "--sentence-silence-duration",
        type=float,
        default=0.30,
        help="Silence in seconds after a sentence-ending delimiter",
    )
    parser.add_argument(
        "--startup-buffer-ms",
        type=float,
        default=160.0,
        help="Minimum real audio accumulated before the first response chunk",
    )
    parser.add_argument("--stall-timeout", type=float, default=DEFAULT_STALL_TIMEOUT_SECONDS)
    parser.add_argument(
        "--synthesis-timeout",
        type=float,
        default=DEFAULT_SYNTHESIS_TIMEOUT_SECONDS,
        help="Maximum seconds allowed for one synthesis request",
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
        help="Seconds to wait for synthesis workers during shutdown",
    )
    parser.add_argument(
        "--max-active-requests",
        type=int,
        default=DEFAULT_MAX_ACTIVE_REQUESTS,
        help="Maximum concurrent synthesis requests",
    )
    parser.add_argument(
        "--max-queued-requests",
        type=int,
        default=DEFAULT_MAX_QUEUED_REQUESTS,
        help="Maximum requests waiting behind active synthesis",
    )
    parser.add_argument(
        "--output-queue-chunks",
        type=int,
        default=DEFAULT_OUTPUT_QUEUE_CHUNKS,
        help="Maximum buffered PCM chunks per request",
    )
    parser.add_argument(
        "--max-output-bytes",
        type=int,
        default=DEFAULT_MAX_OUTPUT_BYTES,
        help="Maximum generated PCM bytes per request",
    )
    parser.add_argument("--log-level", default="info")
    return parser


def _resolve_api_key(args: Any, parser: argparse.ArgumentParser) -> Optional[str]:
    if args.api_key and args.api_key_file:
        parser.error("--api-key and --api-key-file are mutually exclusive")
    api_key = str(args.api_key or os.environ.get("REALTIMETTS_API_KEY") or "").strip()
    if args.api_key_file is not None:
        try:
            api_key = args.api_key_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            parser.error(f"cannot read --api-key-file: {exc}")
        if not api_key:
            parser.error("--api-key-file is empty")
    return api_key or None


def _validate_bind_security(
    args: Any,
    api_key: Optional[str],
    parser: argparse.ArgumentParser,
) -> None:
    if not _is_loopback_host(args.host) and not args.allow_lan:
        parser.error("--allow-lan is required when --host is not loopback")
    if not _is_loopback_host(args.host) and not api_key:
        parser.error(
            "a non-loopback host requires --allow-lan and an API key from "
            "--api-key, --api-key-file, or REALTIMETTS_API_KEY"
        )


def _warn_for_compressed_language_id_model(model_path: Optional[Path]) -> None:
    if model_path is not None and model_path.suffix.lower() == ".ftz":
        LOGGER.warning(
            "Compressed fastText language-ID model %s selected; use the full "
            "lid.176.bin model for production routing accuracy and speed",
            model_path,
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.onset_silence_recovery and args.device != "cpu":
        parser.error("--onset-silence-recovery requires --device cpu")
    if args.cpu_threads is not None and (
        args.device != "cpu" or not 1 <= args.cpu_threads <= 256
    ):
        parser.error("--cpu-threads requires --device cpu and a value between 1 and 256")
    if bool(args.model) != bool(args.codec):
        parser.error("--model and --codec must be supplied together")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.fragment_lookahead_words < 0:
        parser.error("--fragment-lookahead-words must not be negative")
    if args.startup_warmup_tokens <= 0:
        parser.error("--startup-warmup-tokens must be positive")
    if (
        args.synthesis_timeout <= 0
        or args.shutdown_timeout <= 0
        or args.max_active_requests != 1
        or args.max_queued_requests < 0
        or args.output_queue_chunks <= 0
        or args.max_output_bytes <= 0
    ):
        parser.error(
            "max active requests must be 1; timeouts and request/output limits "
            "must be positive; max queued requests may be zero"
        )
    if (
        args.silence_threshold < 0
        or args.trim_pre_roll_ms < 0
        or args.trim_fade_in_ms < 0
        or args.fragment_fade_out_after_ms < 0
        or args.startup_buffer_ms < 0
        or args.comma_silence_duration < 0
        or args.sentence_silence_duration < 0
    ):
        parser.error("silence trimming and startup buffer values must not be negative")
    if any(str(origin).strip() == "*" for origin in (args.cors_origin or [])):
        parser.error("wildcard CORS is not supported; use explicit --cors-origin values")
    api_key = _resolve_api_key(args, parser)
    _validate_bind_security(args, api_key, parser)
    voice_routes: dict[str, dict[str, str]] = {}
    for value in args.voice_language_route:
        try:
            left, target = value.split("=", 1)
            base, route_language = left.rsplit(":", 1)
            if not base.strip() or not route_language.strip() or not target.strip():
                raise ValueError
        except ValueError:
            parser.error("--voice-language-route must be BASE:LANG=VOICE")
        voice_routes.setdefault(base.strip(), {})[route_language.strip()] = target.strip()
    if voice_routes and args.language_id_model is None and str(args.lang).lower() == "auto":
        parser.error("Auto voice-language routes require --language-id-model")
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _warn_for_compressed_language_id_model(args.language_id_model)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - guarded by installation extra
        raise SystemExit(
            "Uvicorn is missing. Install with: pip install \"realtimetts[qwen-server]\""
        ) from exc

    engine_class = QwenCpuEngine if args.device == "cpu" else QwenEngine
    from urllib.parse import urlsplit
    studio_peers = []
    for entry in args.studio_peer:
        label, separator, url = entry.partition("=")
        parsed = urlsplit(url)
        if not separator or not label.strip() or parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            parser.error("--studio-peer requires LABEL=http(s)://host:port without credentials or paths")
        studio_peers.append({"label": label.strip(), "url": url.rstrip("/")})
    device_options = (
        {
            "cpu_threads": args.cpu_threads,
            "onset_silence_recovery": args.onset_silence_recovery,
        }
        if args.device == "cpu"
        else {}
    )
    engine = engine_class(
        model_id=args.model_id,
        quant=args.quant,
        talker_path=args.model,
        codec_path=args.codec,
        model_cache_dir=args.model_cache_dir,
        voice_cache_dir=args.voice_dir / "encoded-cache",
        local_files_only=args.local_files_only,
        library_path=args.library_path,
        use_fa=not args.no_fa,
        clamp_fp16=(args.device != "cpu") if args.clamp_fp16 is None else args.clamp_fp16,
        codec_chunk_sec=args.codec_chunk_sec,
        trim_silence=args.trim_silence,
        silence_threshold=args.silence_threshold,
        trim_pre_roll_ms=args.trim_pre_roll_ms,
        trim_fade_in_ms=args.trim_fade_in_ms,
        fragment_fade_out_after_ms=args.fragment_fade_out_after_ms,
        startup_buffer_ms=args.startup_buffer_ms,
        onset_silence_profile=args.onset_silence_profile,
        clone_mode=args.clone_mode or ("speaker_only" if args.device == "cpu" else "auto"),
        warmup=False,
        warmup_text=args.startup_warmup_text,
        warmup_tokens=args.startup_warmup_tokens,
        **device_options,
    )
    language_router = None
    if args.language_id_model is not None:
        language_router = QwenLanguageRouter(
            FastTextLanguageDetector(args.language_id_model),
            voice_routes=voice_routes,
        )
    server = QwenHttpServer(
        engine,
        alias=args.alias,
        language=args.lang,
        voice_dir=args.voice_dir,
        startup_warmup_voice=args.startup_warmup_voice,
        startup_warmup_routes=args.startup_warmup_routes,
        stall_timeout_seconds=args.stall_timeout,
        synthesis_timeout_seconds=args.synthesis_timeout,
        shutdown_timeout_seconds=args.shutdown_timeout,
        max_active_requests=args.max_active_requests,
        max_queued_requests=args.max_queued_requests,
        output_queue_chunks=args.output_queue_chunks,
        max_output_bytes=args.max_output_bytes,
        api_key=api_key,
        cors_origins=args.cors_origin,
        studio_peers=studio_peers,
        language_router=language_router,
        fragment_lookahead_words=args.fragment_lookahead_words,
        comma_silence_duration=args.comma_silence_duration,
        sentence_silence_duration=args.sentence_silence_duration,
    )
    LOGGER.info(
        "Loaded RealtimeTTS QwenEngine (native=%s, ABI=%s); listening on %s:%s",
        engine.native_version,
        engine.native_abi_version,
        args.host,
        args.port,
    )
    app = create_app(server)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=str(args.log_level).lower(),
        workers=1,
        timeout_keep_alive=5,
    )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "AUDIBLE_AVERAGE_ABS_THRESHOLD",
    "ApiError",
    "QwenHttpServer",
    "RequestMetrics",
    "VoiceRegistry",
    "create_app",
    "main",
]
