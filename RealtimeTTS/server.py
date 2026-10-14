"""Optional generic FastAPI host for a caller-supplied RealtimeTTS engine.

The host deliberately owns no model, voice registry, or engine lifecycle.  A
caller supplies an already configured engine and, when fallback alignment is
requested, an already configured aligner (or an engine native aligner).
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager, suppress
import base64
import binascii
import io
import json
import logging
import queue
import secrets
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import _audio_formats as formats
from .alignment.audio import audio_bytes_to_mono_float32
from .alignment.controller import AlignmentController
from .alignment.timeline import AlignmentConfig

try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, Response
except ImportError:  # pragma: no cover - optional server dependency
    FastAPI = None  # type: ignore[assignment]
    Request = Any  # type: ignore[misc,assignment]
    WebSocket = Any  # type: ignore[misc,assignment]
    WebSocketDisconnect = Exception  # type: ignore[assignment]
    CORSMiddleware = None  # type: ignore[assignment]
    FileResponse = JSONResponse = Response = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

OUTPUT_SAMPLE_RATE = 24_000
OUTPUT_CHANNELS = 1
OUTPUT_ENCODING = "s16le"
OUTPUT_BITS_PER_SAMPLE = 16
PROTOCOL = "realtimetts-generic-http"
PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_STREAM_EVENT_BYTES = 64 * 1024
MAX_STREAM_TEXT_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_AUDIO_QUEUE_CHUNKS = 128
MAX_EVENT_QUEUE = 256
MAX_TEXT_QUEUE = 8
SYNTHESIS_TIMEOUT_SECONDS = 120.0
_ALIGNMENT_FINAL_TIMEOUT = 30.0
_STREAMING_END = object()


class ServerError(Exception):
    def __init__(self, status: int, error_type: str, message: str):
        super().__init__(message)
        self.status = int(status)
        self.error_type = str(error_type)
        self.message = str(message)


def _error_payload(error_type: str, message: str) -> dict[str, Any]:
    return {"error": {"type": error_type, "message": message}}


def _safe_name(value: Any, fallback: str = "generic") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _json_error(exc: ServerError) -> Any:
    return JSONResponse(
        status_code=exc.status,
        content=_error_payload(exc.error_type, exc.message),
    )


def _decode_bearer_subprotocol(protocols: Any) -> Optional[str]:
    for protocol in protocols or ():
        if not isinstance(protocol, str) or not protocol.startswith("bearer."):
            continue
        encoded = protocol[7:]
        try:
            raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            return raw.decode("utf-8")
        except (ValueError, UnicodeDecodeError, binascii.Error):
            return None
    return None


def _pcm_to_wav(pcm: bytes) -> bytes:
    """Write a browser-readable IEEE-float WAVE file."""
    data_size = len(pcm)
    byte_rate = OUTPUT_SAMPLE_RATE * OUTPUT_CHANNELS * 2
    block_align = OUTPUT_CHANNELS * 2
    riff_size = 4 + (8 + 16) + (8 + data_size)
    header = bytearray()
    header.extend(b"RIFF")
    header.extend(int(riff_size).to_bytes(4, "little"))
    header.extend(b"WAVE")
    header.extend(b"fmt ")
    header.extend((16).to_bytes(4, "little"))
    header.extend((1).to_bytes(2, "little"))  # WAVE_FORMAT_PCM
    header.extend(OUTPUT_CHANNELS.to_bytes(2, "little"))
    header.extend(OUTPUT_SAMPLE_RATE.to_bytes(4, "little"))
    header.extend(byte_rate.to_bytes(4, "little"))
    header.extend(block_align.to_bytes(2, "little"))
    header.extend(OUTPUT_BITS_PER_SAMPLE.to_bytes(2, "little"))
    header.extend(b"data")
    header.extend(data_size.to_bytes(4, "little"))
    return bytes(header) + pcm


class _OutboundBuffer:
    """Bounded FIFO with one overflow slot per terminal event type."""

    _TERMINAL_EVENTS = frozenset({"done", "error", "cancelled", "alignment_done"})

    def __init__(self, cancelled: threading.Event, limit: int = MAX_EVENT_QUEUE + MAX_AUDIO_QUEUE_CHUNKS):
        self._cancelled = cancelled
        self._limit = max(1, int(limit))
        self._items: deque[tuple[str, Any]] = deque()
        self._condition = threading.Condition()

    def event(self, payload: dict[str, Any]) -> bool:
        event = dict(payload)
        with self._condition:
            if event.get("type") in self._TERMINAL_EVENTS:
                # Terminal events must not block the event loop or be lost on
                # cancellation. Coalesce duplicates only under queue pressure
                # so the reserve stays bounded without dropping queued PCM.
                if len(self._items) >= self._limit:
                    for index, (kind, pending) in enumerate(self._items):
                        if kind == "event" and pending.get("type") == event["type"]:
                            self._items[index] = ("event", event)
                            return True
            else:
                deadline = time.monotonic() + 0.2
                while len(self._items) >= self._limit:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(remaining)
            self._items.append(("event", event))
            self._condition.notify_all()
            return True

    def audio(self, payload: bytes) -> bool:
        payload = bytes(payload)
        with self._condition:
            while len(self._items) >= self._limit and not self._cancelled.is_set():
                self._condition.wait(0.05)
            if self._cancelled.is_set():
                return False
            self._items.append(("audio", payload))
            self._condition.notify_all()
            return True

    def get(self, timeout: float = 0.1) -> tuple[str, Any]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._items:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                self._condition.wait(remaining)
            item = self._items.popleft()
            self._condition.notify_all()
            return item

    def empty(self) -> bool:
        with self._condition:
            return not self._items


class _BoundedAudioSink:
    def __init__(
        self,
        cancelled: threading.Event,
        *,
        max_chunks: int = MAX_AUDIO_QUEUE_CHUNKS,
        max_bytes: int = MAX_OUTPUT_BYTES,
        ordered: Optional[_OutboundBuffer] = None,
    ):
        self._cancelled = cancelled
        self._ordered = ordered
        self._queue: Optional[queue.Queue[bytes]] = (
            None if ordered is not None else queue.Queue(maxsize=max_chunks)
        )
        self._max_bytes = int(max_bytes)
        self._received_bytes = 0
        self._lock = threading.Lock()
        self.overflow = False

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:
        del block, timeout
        try:
            payload = bytes(item)
        except Exception as exc:
            raise TypeError("engine audio chunks must be bytes-like") from exc
        if not payload:
            return
        with self._lock:
            if self._received_bytes + len(payload) > self._max_bytes:
                self.overflow = True
                self._cancelled.set()
                return
            self._received_bytes += len(payload)
        if self._ordered is not None:
            self._ordered.audio(payload)
            return
        assert self._queue is not None
        while not self._cancelled.is_set():
            try:
                self._queue.put(payload, timeout=0.05)
                return
            except queue.Full:
                continue

    def put_nowait(self, item: Any) -> None:
        self.put(item, block=False)

    def get(self, timeout: float = 0.1) -> bytes:
        if self._queue is None:
            raise queue.Empty
        return self._queue.get(timeout=timeout)

    def empty(self) -> bool:
        return True if self._queue is None else self._queue.empty()

    def qsize(self) -> int:
        return 0 if self._queue is None else self._queue.qsize()


class _BoundedTimingSink:
    """Drop old native timing records instead of blocking synthesis."""

    def __init__(self, limit: int = MAX_EVENT_QUEUE):
        self._items: deque[Any] = deque(maxlen=limit)
        self._lock = threading.Lock()

    def put(self, item: Any, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        with self._lock:
            self._items.append(item)

    def put_nowait(self, item: Any) -> None:
        self.put(item)

    def qsize(self) -> int:
        with self._lock:
            return len(self._items)


class _EventBuffer:
    def __init__(self, limit: int = MAX_EVENT_QUEUE):
        self._items: deque[dict[str, Any]] = deque(maxlen=limit)
        self._condition = threading.Condition()
        self._limit = int(limit)

    def put(self, event: dict[str, Any]) -> None:
        with self._condition:
            if len(self._items) >= self._limit:
                self._items.popleft()
            self._items.append(dict(event))
            self._condition.notify()

    def get(self, timeout: float = 0.1) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._items:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                self._condition.wait(remaining)
            return self._items.popleft()

    def empty(self) -> bool:
        with self._condition:
            return not self._items


class _PCMConverter:
    def __init__(self):
        self._resampler = None
        self._source_rate: Optional[int] = None

    def _get_resampler(self, source_rate: int):
        if self._resampler is not None and self._source_rate == source_rate:
            return self._resampler
        try:
            import av
        except ImportError as exc:
            raise ServerError(
                503,
                "server_error",
                "PyAV is required to resample engine audio to 24 kHz",
            ) from exc
        self._resampler = av.AudioResampler(
            format="flt", layout="mono", rate=OUTPUT_SAMPLE_RATE
        )
        self._source_rate = source_rate
        return self._resampler

    @staticmethod
    def _pack(samples: Any) -> bytes:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(values):
            return b""
        values = np.clip(values, -1.0, 1.0)
        return np.rint(values * 32767.0).astype("<i2", copy=False).tobytes()

    def convert(self, payload: bytes, stream_format: tuple[int, int, int]) -> bytes:
        fmt, channels, sample_rate = stream_format
        samples, source_rate = audio_bytes_to_mono_float32(
            payload, fmt, channels, sample_rate
        )
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(samples):
            return b""
        if int(source_rate) == OUTPUT_SAMPLE_RATE:
            return self._pack(samples)
        try:
            import av
        except ImportError as exc:
            raise ServerError(
                503,
                "server_error",
                "PyAV is required to resample engine audio to 24 kHz",
            ) from exc
        resampler = self._get_resampler(int(source_rate))
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1), format="flt", layout="mono"
        )
        frame.sample_rate = int(source_rate)
        output = []
        for decoded in resampler.resample(frame):
            output.append(decoded.to_ndarray().reshape(-1))
        return b"".join(self._pack(part) for part in output)

    def finish(self) -> bytes:
        if self._resampler is None:
            return b""
        output = []
        for decoded in self._resampler.resample(None):
            output.append(decoded.to_ndarray().reshape(-1))
        self._resampler = None
        self._source_rate = None
        return b"".join(self._pack(part) for part in output)


class _DecodeQueue:
    """Decode compressed engine chunks when alignment is disabled."""

    def __init__(self, wrapped: _BoundedAudioSink, stream_format: tuple[int, int, int]):
        from .alignment.decoder import MP3Decoder

        rate = stream_format[2] if stream_format[2] > 0 else OUTPUT_SAMPLE_RATE
        self._decoder = MP3Decoder(rate)
        self._wrapped = wrapped

    def put(self, item: Any, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        for chunk in self._decoder.feed(bytes(item)):
            self._wrapped.put(chunk)

    def put_nowait(self, item: Any) -> None:
        self.put(item)

    def finish(self) -> None:
        for chunk in self._decoder.finish():
            self._wrapped.put(chunk)


def _span_payload(span: Any, level: str) -> dict[str, Any]:
    return span.to_dict(alias="character" if level == "character" else "word")


class _GenericSession:
    def __init__(
        self,
        host: "GenericServer",
        options: dict[str, Any],
        alignment_config: Optional[AlignmentConfig],
        *,
        session_id: Optional[str] = None,
        ordered: bool = False,
    ):
        self.host = host
        self.engine = host.engine
        self.options = dict(options)
        self.alignment_config = alignment_config
        self.session_id = session_id or uuid.uuid4().hex
        self.request_id = _safe_name(self.options.get("request_id"), self.session_id)
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.audio_done = threading.Event()
        self.events = _EventBuffer()
        self.outbound: Optional[_OutboundBuffer] = (
            _OutboundBuffer(self.cancelled) if ordered else None
        )
        self.audio = _BoundedAudioSink(
            self.cancelled,
            max_bytes=host.max_output_bytes,
            ordered=self.outbound,
        )
        self.timings = _BoundedTimingSink()
        self.items: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=MAX_TEXT_QUEUE)
        self.total_text_bytes = 0
        self.fragment_texts: dict[int, str] = {}
        self.fragment_audio_starts: dict[int, float] = {}
        self.fragment_text_starts: dict[int, int] = {}
        self.controller: Optional[AlignmentController] = None
        self.binding = None
        self.original_queue = None
        self.original_timings = None
        self.decode_queue: Optional[_DecodeQueue] = None
        self.output_format: tuple[int, int, int] = (
            formats.INT16, 1, OUTPUT_SAMPLE_RATE
        )
        self.error: Optional[ServerError] = None
        self._worker = threading.Thread(
            target=self._run,
            name=f"realtimetts-generic-{self.session_id[:8]}",
            daemon=True,
        )
        self._engine_locked = False
        self._ended = False
        self._lifecycle_lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        event.setdefault("session_id", self.session_id)
        event.setdefault("request_id", self.request_id)
        if self.outbound is not None:
            self.outbound.event(event)
        else:
            self.events.put(event)

    def start(self) -> None:
        self._worker.start()

    def add_text(self, text: str) -> None:
        if self._ended:
            raise ServerError(400, "invalid_request_error", "text arrived after end")
        if not isinstance(text, str) or not text.strip():
            raise ServerError(400, "invalid_request_error", "text must be non-empty")
        encoded = text.encode("utf-8")
        if self.total_text_bytes + len(encoded) > MAX_STREAM_TEXT_BYTES:
            raise ServerError(413, "invalid_request_error", "stream text is too large")
        self.total_text_bytes += len(encoded)
        try:
            self.items.put(("text", text), timeout=0.2)
        except queue.Full as exc:
            raise ServerError(429, "rate_limit_error", "stream text queue is full") from exc

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        while not self.cancelled.is_set():
            try:
                self.items.put(("end", ""), timeout=0.05)
                return
            except queue.Full:
                continue

    def cancel(self) -> None:
        with self._lifecycle_lock:
            if self.cancelled.is_set():
                return
            self.cancelled.set()
            # A rejected or completed session must not stop another request.
            if not self._engine_locked:
                return
            stop = getattr(self.engine, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    LOGGER.warning("Generic engine stop failed", exc_info=True)

    def pause(self) -> bool:
        method = getattr(self.engine, "pause", None)
        if not callable(method):
            return False
        result = method()
        return result is not False

    def resume(self) -> bool:
        method = getattr(self.engine, "resume", None)
        if not callable(method):
            return False
        result = method()
        return result is not False

    def _alignment_for_snapshot(self, snapshot: Any) -> list[dict[str, Any]]:
        grouped: dict[int, dict[str, Any]] = {}
        for level, spans in (("word", snapshot.words), ("character", snapshot.characters)):
            for span in spans:
                parts = str(span.fragment_id).rsplit(":", 1)
                try:
                    index = int(parts[-1]) - 1
                except (ValueError, TypeError):
                    continue
                if index < 0:
                    continue
                payload = grouped.setdefault(index, {
                    "fragment_index": index,
                    "text": self.fragment_texts.get(index, ""),
                    "words": [],
                    "characters": [],
                    "status": snapshot.status,
                    "final": snapshot.final,
                    "revision": snapshot.revision,
                    "source": span.source,
                    "sample_rate_hz": OUTPUT_SAMPLE_RATE,
                    "time_base": "fragment",
                })
                payload["source"] = span.source
                payload["status"] = snapshot.status
                payload["final"] = snapshot.final
                payload["revision"] = snapshot.revision
                payload["sample_rate_hz"] = OUTPUT_SAMPLE_RATE
                payload["time_base"] = "fragment"
                item = _span_payload(span, level)
                audio_offset = self.fragment_audio_starts.get(index, 0.0)
                item["start_time"] = max(0.0, float(item["start_time"]) - audio_offset)
                item["end_time"] = max(item["start_time"], float(item["end_time"]) - audio_offset)
                text_offset = self.fragment_text_starts.get(index, 0)
                if item.get("text_start") is not None:
                    item["text_start"] = max(0, int(item["text_start"]) - text_offset)
                if item.get("text_end") is not None:
                    item["text_end"] = max(0, int(item["text_end"]) - text_offset)
                payload["words" if level == "word" else "characters"].append(item)
        for index, text in self.fragment_texts.items():
            grouped.setdefault(index, {
                "fragment_index": index,
                "text": text,
                "words": [],
                "characters": [],
                "status": snapshot.status,
                "final": snapshot.final,
                "revision": snapshot.revision,
                "source": "unknown",
                "sample_rate_hz": OUTPUT_SAMPLE_RATE,
                "time_base": "fragment",
            })
        return [grouped[index] for index in sorted(grouped)]

    def _on_alignment(self, snapshot: Any) -> None:
        for fragment in self._alignment_for_snapshot(snapshot):
            self.emit({"type": "alignment", **fragment})

    def _prepare_engine(self) -> None:
        with self._lifecycle_lock:
            if not self.host._engine_lock.acquire(blocking=False):
                raise ServerError(429, "rate_limit_error", "engine is busy")
            self._engine_locked = True
        engine = self.engine
        self.original_queue = engine.queue
        self.original_timings = getattr(engine, "timings", _BoundedTimingSink())
        input_format = tuple(engine.get_stream_info())
        if len(input_format) != 3:
            raise ServerError(500, "server_error", "engine get_stream_info() must return (format, channels, rate)")
        if self.alignment_config is not None:
            self.controller = AlignmentController(
                self.alignment_config, on_update=self._on_alignment
            )
            engine.queue = self.audio
            engine.timings = self.timings
            self.binding = self.controller.bind(engine)
            self.output_format = self.binding.output_format
        else:
            engine.queue = self.audio
            engine.timings = self.timings
            if input_format[0] == formats.CUSTOM:
                self.decode_queue = _DecodeQueue(self.audio, input_format)
                engine.queue = self.decode_queue
                self.output_format = (formats.FLOAT32, 1, OUTPUT_SAMPLE_RATE)
            else:
                self.output_format = input_format

    def _apply_options(self) -> None:
        voice = self.options.get("voice")
        if isinstance(voice, str) and voice.strip() and voice.strip() != "default":
            setter = getattr(self.engine, "set_voice", None)
            if callable(setter):
                try:
                    setter(voice.strip())
                except Exception as exc:
                    raise ServerError(400, "invalid_request_error", str(exc)) from exc
        parameters = {
            key: value
            for key, value in self.options.items()
            if key in {
                "temperature", "top_k", "top_p", "seed", "max_new_tokens",
                "repetition_penalty", "do_sample", "cfg_scale",
            }
        }
        setter = getattr(self.engine, "set_voice_parameters", None)
        if parameters and callable(setter):
            try:
                setter(**parameters)
            except (NotImplementedError, AttributeError):
                pass
            except Exception as exc:
                raise ServerError(400, "invalid_request_error", str(exc)) from exc

    def _synthesize_fragment(self, text: str, index: int) -> None:
        self.fragment_texts[index] = text
        if self.controller is not None:
            self.controller.append_source(text)
            self.emit({
                "type": "fragment_start",
                "fragment_index": index,
                "text": text,
                "source": self.alignment_config.source,
                "status": "aligning",
                "final": False,
                "revision": self.controller.snapshot().revision,
                "sample_rate_hz": OUTPUT_SAMPLE_RATE,
                "time_base": "fragment",
            })
            self.binding.begin(text)
            fragment = self.binding.fragment
            self.fragment_audio_starts[index] = float(getattr(fragment, "start", 0.0) or 0.0)
            self.fragment_text_starts[index] = int(getattr(fragment, "text_offset", 0) or 0)
        ok = False
        try:
            synthesize = getattr(self.engine, "synthesize", None)
            if not callable(synthesize):
                raise ServerError(500, "server_error", "engine does not implement synthesize(text)")
            ok = bool(synthesize(text))
            if not ok:
                raise ServerError(502, "server_error", "engine synthesis failed")
        finally:
            if self.binding is not None:
                try:
                    self.binding.finish(success=ok and not self.cancelled.is_set())
                except Exception:
                    if not self.cancelled.is_set():
                        raise
        if self.audio.overflow:
            raise ServerError(503, "output_error", "synthesis output exceeded the configured limit")
        if self.controller is not None:
            self.emit({
                "type": "fragment_end",
                "fragment_index": index,
                "status": "aligning",
                "final": False,
                "revision": self.controller.snapshot().revision,
                "sample_rate_hz": OUTPUT_SAMPLE_RATE,
                "time_base": "fragment",
            })

    def _finish_alignment(self) -> None:
        if self.controller is None:
            return
        self.controller.finish()
        try:
            snapshot = self.controller.wait(timeout=_ALIGNMENT_FINAL_TIMEOUT)
        except TimeoutError:
            snapshot = self.controller.snapshot()
        fragments = self._alignment_for_snapshot(snapshot)
        self.emit({
            "type": "alignment_done",
            "complete": bool(snapshot.final and snapshot.status == "ready" and not snapshot.errors),
            "coverage": snapshot.coverage,
            "status": snapshot.status,
            "final": bool(snapshot.final),
            "revision": snapshot.revision,
            "fragments": fragments,
            "sample_rate_hz": OUTPUT_SAMPLE_RATE,
            "time_base": "fragment",
        })

    def collect_audio(self) -> bytes:
        converter = _PCMConverter()
        chunks: list[bytes] = []
        deadline = time.monotonic() + SYNTHESIS_TIMEOUT_SECONDS
        while not self.audio_done.is_set() or not self.audio.empty():
            if time.monotonic() >= deadline:
                self.cancel()
                raise ServerError(504, "timeout_error", "synthesis timed out")
            try:
                raw = self.audio.get(0.05)
            except queue.Empty:
                continue
            payload = converter.convert(raw, self.output_format)
            if payload:
                chunks.append(payload)
        tail = converter.finish()
        if tail:
            chunks.append(tail)
        self.done.wait(max(0.0, deadline - time.monotonic()))
        return b"".join(chunks)

    def _cleanup(self) -> None:
        if self.decode_queue is not None:
            try:
                self.decode_queue.finish()
            except Exception:
                if not self.cancelled.is_set():
                    LOGGER.warning("Compressed audio flush failed", exc_info=True)
        if self.binding is not None:
            with suppress(Exception):
                self.binding.detach()
        if self.original_queue is not None:
            self.engine.queue = self.original_queue
        if self.original_timings is not None:
            self.engine.timings = self.original_timings
        if self.controller is not None:
            with suppress(Exception):
                self.controller.close()
        self.audio_done.set()
        with self._lifecycle_lock:
            if self._engine_locked:
                self._engine_locked = False
                self.host._engine_lock.release()

    def _run(self) -> None:
        try:
            self._prepare_engine()
            reset_duration = getattr(self.engine, "reset_audio_duration", None)
            if callable(reset_duration):
                reset_duration()
            self._apply_options()
            while not self.cancelled.is_set():
                try:
                    kind, text = self.items.get(timeout=0.1)
                except queue.Empty:
                    continue
                if kind == "end":
                    break
                self._synthesize_fragment(text, len(self.fragment_texts))
            if not self.cancelled.is_set():
                self._finish_alignment()
        except ServerError as exc:
            if self.cancelled.is_set() and exc.error_type != "output_error":
                self.error = None
            else:
                self.error = exc
                if not self.cancelled.is_set() or exc.error_type == "output_error":
                    self.emit({
                        "type": "error",
                        "message": exc.message,
                        "error": {"type": exc.error_type, "message": exc.message},
                    })
        except Exception as exc:
            if self.cancelled.is_set():
                self.error = None
            else:
                self.error = ServerError(500, "server_error", str(exc) or exc.__class__.__name__)
                LOGGER.exception("Generic synthesis failed")
                self.emit({
                    "type": "error",
                    "message": self.error.message,
                    "error": {"type": self.error.error_type, "message": self.error.message},
                })
        finally:
            if self.cancelled.is_set() and not self.error:
                self.emit({"type": "cancelled", "status": "cancelled", "final": True})
            if not self.error and not self.cancelled.is_set():
                self.emit({"type": "done", "status": "ready", "final": True})
            self._cleanup()
            self.done.set()


class GenericServer:
    def __init__(
        self,
        engine: Any,
        *,
        aligner: Any = None,
        api_key: Optional[str] = None,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
    ):
        if engine is None:
            raise ValueError("engine is required")
        for method in ("get_stream_info", "synthesize"):
            if not callable(getattr(engine, method, None)):
                raise TypeError(f"engine must implement {method}()")
        if not hasattr(engine, "queue"):
            raise TypeError("engine must expose queue")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.engine = engine
        self.aligner = aligner
        self.api_key = str(api_key) if api_key else None
        self.max_output_bytes = int(max_output_bytes)
        self._engine_lock = threading.Lock()
        self._ready = True

    def authorized(self, authorization: Optional[str], query_key: Optional[str] = None) -> bool:
        token = ""
        if isinstance(authorization, str) and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if not token and isinstance(query_key, str):
            token = query_key
        return self.api_key is None or bool(token) and secrets.compare_digest(token, self.api_key)

    def alignment_config(self, value: Any, language: Optional[str] = None) -> Optional[AlignmentConfig]:
        if isinstance(value, dict):
            allowed = {"source", "mode", "level", "max_buffer_seconds"}
            unknown = sorted(set(value) - allowed)
            if unknown:
                raise ServerError(
                    400,
                    "invalid_request_error",
                    "alignment fields not allowed over the server API: " + ", ".join(unknown),
                )
            if "max_buffer_seconds" in value:
                try:
                    if float(value["max_buffer_seconds"]) > 1.0:
                        raise ServerError(
                            400,
                            "invalid_request_error",
                            "alignment max_buffer_seconds must be at most 1.0",
                        )
                except (TypeError, ValueError) as exc:
                    raise ServerError(
                        400,
                        "invalid_request_error",
                        "alignment max_buffer_seconds must be a finite number",
                    ) from exc
        try:
            config = AlignmentConfig.coerce(value, aligner=self.aligner, language=language)
        except (TypeError, ValueError) as exc:
            raise ServerError(400, "invalid_request_error", str(exc)) from exc
        if config is None:
            return None
        if config.model_dir is not None:
            raise ServerError(
                400,
                "unsupported_operation",
                "model_dir is not accepted by the server API; supply an aligner to create_app",
            )
        streaming = getattr(self.engine, "streaming_aligner", None)
        native = bool(getattr(self.engine, "provides_word_timings", False)) and (
            config.level == "word"
            or "character" in getattr(streaming, "timestamp_levels", ())
        )
        if config.source == "native" and not native:
            raise ServerError(
                400,
                "unsupported_operation",
                "the engine does not provide the requested native timestamps",
            )
        if config.source != "model" and native:
            return config
        if config.aligner is None and streaming is None:
            raise ServerError(
                400,
                "unsupported_operation",
                "alignment requires a supplied aligner, or engine.streaming_aligner",
            )
        return config

    def voices(self) -> list[dict[str, Any]]:
        getter = getattr(self.engine, "get_voices", None)
        values = []
        if callable(getter):
            try:
                values = list(getter() or [])
            except NotImplementedError:
                values = []
        result = []
        for value in values:
            name = getattr(value, "name", value)
            if isinstance(name, str) and name.strip():
                result.append({"name": name.strip(), "kind": "engine"})
        if not result:
            result.append({"name": "default", "kind": "engine_default"})
        return result

    def capabilities(self) -> dict[str, Any]:
        engine_name = _safe_name(getattr(self.engine, "engine_name", None), "generic")
        model_id = _safe_name(
            getattr(self.engine, "model_id", None),
            engine_name,
        )
        streaming = getattr(self.engine, "streaming_aligner", None)
        native_word = bool(getattr(self.engine, "provides_word_timings", False))
        alignment = self.aligner is not None or streaming is not None or native_word
        native_pause = callable(getattr(self.engine, "pause", None)) and callable(
            getattr(self.engine, "resume", None)
        )
        return {
            "object": "capabilities",
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "server_version": "generic",
            "model": {
                "id": model_id,
                "model_id": model_id,
                "object": "model",
                "owned_by": "caller",
            },
            "engine": {
                "name": engine_name,
                "backend": _safe_name(getattr(self.engine, "backend", None), "BaseEngine"),
                "device": _safe_name(getattr(self.engine, "device", None), "unknown"),
                "quant": _safe_name(getattr(self.engine, "quant", None), "unknown"),
                "model_id": model_id,
            },
            "audio": {
                "encoding": OUTPUT_ENCODING,
                "sample_rate_hz": OUTPUT_SAMPLE_RATE,
                "channels": OUTPUT_CHANNELS,
                "bits_per_sample": OUTPUT_BITS_PER_SAMPLE,
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
                "input_events": ["config", "text", "end", "cancel", "flush"]
                + (["pause", "resume"] if native_pause else []),
                "segmented_narration": False,
                "native_pause_resume": native_pause,
                "pause_acknowledgement": "engine_method" if native_pause else "client_playback",
            },
            "limits": {
                "max_request_bytes": MAX_REQUEST_BYTES,
                "max_stream_event_bytes": MAX_STREAM_EVENT_BYTES,
                "max_stream_text_bytes": MAX_STREAM_TEXT_BYTES,
                "max_active_requests": 1,
                "max_queued_requests": 0,
                "output_queue_chunks": MAX_AUDIO_QUEUE_CHUNKS,
                "max_output_bytes": self.max_output_bytes,
                "synthesis_timeout_seconds": SYNTHESIS_TIMEOUT_SECONDS,
            },
            "authentication": {"required": self.api_key is not None, "scheme": "bearer"},
            "sampling_defaults": {},
            "studio_peers": [],
            "features": {
                "model_type": "generic",
                "voice_cloning": False,
                "instruction_control": False,
                "per_text_voice": bool(callable(getattr(self.engine, "set_voice", None))),
                "sentence_splitting": False,
                "text_exchange": False,
                "streaming_alignment": alignment,
                "alignment_timestamps": (
                    list(getattr(streaming, "timestamp_levels", ()))
                    if streaming is not None else (["word"] if native_word else [])
                ),
                "alignment_language": getattr(getattr(streaming, "backend", None), "language", None) if streaming else None,
                "alignment_language_timestamps": getattr(streaming, "language_timestamps", {}) if streaming else {},
                "hallucination_guard": False,
            },
            "ready": self._ready,
        }

    def new_session(
        self,
        payload: dict[str, Any],
        *,
        session_id: Optional[str] = None,
        ordered: bool = False,
    ) -> _GenericSession:
        if not isinstance(payload, dict):
            raise ServerError(400, "invalid_request_error", "request must be a JSON object")
        alignment = self.alignment_config(payload.get("alignment"), payload.get("language"))
        return _GenericSession(
            self, payload, alignment, session_id=session_id, ordered=ordered
        )

    def health(self) -> dict[str, Any]:
        acquired = self._engine_lock.acquire(False)
        if acquired:
            self._engine_lock.release()
        return {
            "object": "health",
            "status": "ok" if self._ready else "starting",
            "ready": self._ready,
            "engine": _safe_name(getattr(self.engine, "engine_name", None), "generic"),
            "active_requests": 0 if acquired else 1,
        }

    def shutdown(self) -> None:
        self._ready = False


def _event_from_text(raw: str) -> dict[str, Any]:
    if len(raw.encode("utf-8")) > MAX_STREAM_EVENT_BYTES:
        raise ServerError(413, "invalid_request_error", "stream event is too large")
    try:
        event = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ServerError(400, "invalid_request_error", "stream event is not valid JSON") from exc
    if not isinstance(event, dict):
        raise ServerError(400, "invalid_request_error", "stream event must be a JSON object")
    return event


async def _request_json(request: Any) -> dict[str, Any]:
    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        raise ServerError(413, "invalid_request_error", "request is too large")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ServerError(400, "invalid_request_error", "request is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ServerError(400, "invalid_request_error", "request must be a JSON object")
    return payload


async def _monitor_http_disconnect(request: Any, session: _GenericSession) -> None:
    while not session.done.is_set():
        try:
            if await request.is_disconnected():
                session.cancel()
                return
        except Exception:
            return
        await asyncio.sleep(0.1)


def create_app(
    engine: Any,
    *,
    aligner: Any = None,
    api_key: Optional[str] = None,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
) -> Any:
    """Create a generic optional FastAPI app around an already loaded engine."""
    if FastAPI is None:  # pragma: no cover
        raise ImportError(
            "The generic RealtimeTTS server requires FastAPI. "
            "Install the project's server extra."
        )
    host = GenericServer(
        engine,
        aligner=aligner,
        api_key=api_key,
        max_output_bytes=max_output_bytes,
    )
    studio_root = Path(__file__).with_name("studio")

    @asynccontextmanager
    async def lifespan(_app: Any):
        host._ready = True
        try:
            yield
        finally:
            host.shutdown()

    app = FastAPI(title="RealtimeTTS generic server", lifespan=lifespan)
    app.state.realtimetts_server = host
    if CORSMiddleware is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost", "http://127.0.0.1", "http://[::1]"],
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.middleware("http")
    async def auth(request: Any, call_next: Any) -> Any:
        if request.url.path.startswith("/v1/") and not host.authorized(
            request.headers.get("authorization"),
            request.query_params.get("api_key"),
        ):
            return JSONResponse(
                status_code=401,
                content=_error_payload("authentication_error", "missing or invalid API key"),
            )
        return await call_next(request)

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
        return host.health()

    @app.get("/ready")
    def ready() -> Any:
        if host._ready:
            return {"status": "ready"}
        return JSONResponse(status_code=503, content={"status": "not_ready"})

    @app.get("/v1/capabilities")
    def capabilities() -> Any:
        return host.capabilities()

    @app.get("/v1/models")
    def models() -> Any:
        caps = host.capabilities()
        return {
            "object": "list",
            "data": [caps["model"]],
        }

    @app.get("/v1/audio/voices")
    def voices() -> Any:
        return {"voices": host.voices()}

    @app.post("/v1/audio/speech")
    async def speech(request: Request) -> Any:
        try:
            payload = await _request_json(request)
            text = payload.get("input")
            if not isinstance(text, str) or not text.strip():
                raise ServerError(400, "invalid_request_error", "'input' must be a non-empty string")
            response_format = payload.get("response_format", "pcm")
            if response_format not in {"pcm", "wav"}:
                raise ServerError(400, "invalid_request_error", "response_format must be 'pcm' or 'wav'")
            alignment = payload.get("alignment")
            if alignment not in (None, False) and not (
                isinstance(alignment, str) and alignment.strip().lower() in {"off", "none"}
            ):
                raise ServerError(
                    400,
                    "unsupported_operation",
                    "alignment metadata requires the WebSocket streaming endpoint",
                )
            session = host.new_session(payload)
            session.start()
            session.add_text(text)
            session.end()
            monitor = asyncio.create_task(_monitor_http_disconnect(request, session))
            try:
                pcm = await asyncio.to_thread(session.collect_audio)
            except asyncio.CancelledError:
                session.cancel()
                await asyncio.to_thread(session.done.wait, 1.0)
                raise
            finally:
                monitor.cancel()
                with suppress(asyncio.CancelledError):
                    await monitor
            if session.error is not None:
                raise session.error
            content = _pcm_to_wav(pcm) if response_format == "wav" else pcm
            media_type = "audio/wav" if response_format == "wav" else "audio/pcm"
            return Response(
                content=content,
                media_type=media_type,
                headers={
                    "Cache-Control": "no-cache",
                    "X-Audio-Encoding": OUTPUT_ENCODING,
                    "X-Audio-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
                    "X-Audio-Channels": str(OUTPUT_CHANNELS),
                    "X-Audio-Bits-Per-Sample": str(OUTPUT_BITS_PER_SAMPLE),
                },
            )
        except ServerError as exc:
            return _json_error(exc)

    @app.websocket("/v1/audio/speech-stream")
    async def speech_stream(websocket: WebSocket) -> None:
        offered = websocket.scope.get("subprotocols", [])
        browser_key = _decode_bearer_subprotocol(offered)
        if not host.authorized(
            websocket.headers.get("authorization"),
            browser_key or websocket.query_params.get("api_key"),
        ):
            await websocket.close(code=1008, reason="missing or invalid API key")
            return
        await websocket.accept(subprotocol="qwen-studio" if "qwen-studio" in offered else None)
        try:
            first = _event_from_text(await websocket.receive_text())
            if first.get("type") != "config":
                raise ServerError(400, "invalid_request_error", "first stream event must be config")
            if first.get("response_format", "pcm") != "pcm":
                raise ServerError(400, "invalid_request_error", "WebSocket response_format must be 'pcm'")
            session = host.new_session(
                first, session_id=uuid.uuid4().hex, ordered=True
            )
        except ServerError as exc:
            await websocket.send_json({
                "type": "error",
                "message": exc.message,
                "error": {"type": exc.error_type, "message": exc.message},
            })
            await websocket.close()
            return

        session.start()
        session.emit({
            "type": "ready",
            "request_id": session.request_id,
            "audio": host.capabilities()["audio"],
        })
        async def send_outbound() -> None:
            assert session.outbound is not None
            converter = _PCMConverter()
            while not session.audio_done.is_set() or not session.outbound.empty():
                try:
                    kind, item = await asyncio.to_thread(session.outbound.get, 0.1)
                except queue.Empty:
                    continue
                if kind == "event":
                    await websocket.send_json(item)
                    continue
                try:
                    payload = converter.convert(item, session.output_format)
                except ServerError as exc:
                    session.error = exc
                    session.cancel()
                    session.emit({
                        "type": "error",
                        "message": exc.message,
                        "error": {"type": exc.error_type, "message": exc.message},
                    })
                    break
                if payload:
                    await websocket.send_bytes(payload)
            tail = converter.finish()
            if tail:
                await websocket.send_bytes(tail)

        async def receive_events() -> None:
            try:
                while not session.done.is_set():
                    event = _event_from_text(await websocket.receive_text())
                    event_type = event.get("type")
                    if event_type == "text":
                        session.add_text(event.get("text"))
                    elif event_type == "end":
                        session.end()
                    elif event_type == "cancel":
                        session.cancel()
                        return
                    elif event_type == "pause":
                        if not session.pause():
                            raise ServerError(400, "unsupported_operation", "engine does not support pause")
                        session.emit({"type": "paused"})
                    elif event_type == "resume":
                        if not session.resume():
                            raise ServerError(400, "unsupported_operation", "engine does not support resume")
                        session.emit({"type": "resumed"})
                    elif event_type == "flush":
                        session.emit({"type": "flushed"})
                    else:
                        raise ServerError(400, "invalid_request_error", f"unknown stream event type: {event_type!r}")
            except WebSocketDisconnect:
                session.cancel()
            except ServerError as exc:
                session.error = exc
                session.cancel()
                session.emit({
                    "type": "error",
                    "message": exc.message,
                    "error": {"type": exc.error_type, "message": exc.message},
                })

        outbound_task = asyncio.create_task(send_outbound())
        receive_task = asyncio.create_task(receive_events())
        done_task = asyncio.create_task(asyncio.to_thread(session.done.wait))
        try:
            await done_task
        finally:
            receive_task.cancel()
            with suppress(asyncio.CancelledError):
                await receive_task
            await asyncio.gather(outbound_task, return_exceptions=True)
            with suppress(Exception):
                await websocket.close()

    return app


__all__ = ["GenericServer", "ServerError", "create_app"]
