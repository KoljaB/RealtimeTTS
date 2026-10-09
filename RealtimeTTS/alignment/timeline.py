"""Thread-safe public alignment data and timeline primitives.

The timeline deliberately owns data and state transitions only. Audio workers,
model scheduling, and word callbacks remain the responsibility of the caller
that feeds this object.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import json
import logging
import math
from pathlib import Path
import re
import threading
import time
from numbers import Integral
from typing import Any, Callable, Optional, Union


__all__ = [
    "AlignedSpan",
    "AlignmentConfig",
    "AlignmentSnapshot",
    "AlignmentTimeline",
    "map_alignment_offsets",
    "map_text_offsets",
    "project_text_offsets",
]


_ALIGNMENT_SOURCES = frozenset({"auto", "native", "model"})
_ALIGNMENT_MODES = frozenset({"live", "synced", "export"})
_ALIGNMENT_LEVELS = frozenset({"word", "character"})
_SENTENCE_END = re.compile(r"[.!?…。！？…]$")


def _finite_float(value: Any, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _label(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _level(value: Any) -> str:
    value = _label(value, "level").lower()
    if value not in _ALIGNMENT_LEVELS:
        raise ValueError(f"level must be one of {sorted(_ALIGNMENT_LEVELS)}")
    return value


def _optional_index(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer or None")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class AlignedSpan:
    """One word or character with a measured position in the audio timeline."""

    text: str
    start_time: float
    end_time: float
    text_start: Optional[int] = None
    text_end: Optional[int] = None
    source: str = "model"
    state: str = "final"
    level: str = "word"
    score: Optional[float] = None
    fragment_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        start = _finite_float(self.start_time, "start_time")
        end = _finite_float(self.end_time, "end_time")
        if end < start:
            raise ValueError("end_time must be greater than or equal to start_time")
        text_start = _optional_index(self.text_start, "text_start")
        text_end = _optional_index(self.text_end, "text_end")
        if text_start is not None and text_end is not None and text_end < text_start:
            raise ValueError("text_end must be greater than or equal to text_start")
        source = _label(self.source, "source")
        state = _label(self.state, "state")
        level = _level(self.level)
        score = None if self.score is None else _finite_float(
            self.score, "score", nonnegative=False
        )
        if not isinstance(self.fragment_id, str):
            raise TypeError("fragment_id must be a string")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "text_start", text_start)
        object.__setattr__(self, "text_end", text_end)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "score", score)

    @property
    def word(self) -> str:
        return self.text

    @property
    def character(self) -> str:
        return self.text

    def to_dict(self, *, alias: Optional[str] = None) -> dict[str, Any]:
        key = alias or ("character" if self.level == "character" else "word")
        return {
            "text": self.text,
            key: self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text_start": self.text_start,
            "text_end": self.text_end,
            "source": self.source,
            "state": self.state,
            "level": self.level,
            "score": self.score,
            "fragment_id": self.fragment_id,
        }


@dataclass(frozen=True)
class AlignmentConfig:
    """Validated configuration shared by live, synced, and export alignment."""

    source: str = "auto"
    mode: str = "live"
    level: str = "word"
    language: Optional[str] = None
    aligner: Any = None
    model_dir: Optional[Union[str, Path]] = None
    max_buffer_seconds: float = 0.25
    update_interval: float = 0.25
    max_pending: int = 8
    max_audio_seconds: float = 120.0

    def __post_init__(self) -> None:
        source = _label(self.source, "source").lower()
        mode = _label(self.mode, "mode").lower()
        level = _level(self.level)
        if source not in _ALIGNMENT_SOURCES:
            raise ValueError(f"source must be one of {sorted(_ALIGNMENT_SOURCES)}")
        if mode not in _ALIGNMENT_MODES:
            raise ValueError(f"mode must be one of {sorted(_ALIGNMENT_MODES)}")
        if self.language is not None:
            if not isinstance(self.language, str) or not self.language.strip():
                raise ValueError("language must be a non-empty string or None")
            language = self.language.strip()
        else:
            language = None
        model_dir = self.model_dir
        if model_dir is not None and not isinstance(model_dir, (str, Path)):
            try:
                model_dir = model_dir.__fspath__()
            except AttributeError as exc:
                raise TypeError("model_dir must be path-like or None") from exc
        max_buffer_seconds = _finite_float(self.max_buffer_seconds, "max_buffer_seconds")
        update_interval = _finite_float(self.update_interval, "update_interval")
        max_audio_seconds = _finite_float(self.max_audio_seconds, "max_audio_seconds")
        if max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be greater than zero")
        if update_interval <= 0:
            raise ValueError("update_interval must be greater than zero")
        if isinstance(self.max_pending, bool) or not isinstance(self.max_pending, Integral):
            raise TypeError("max_pending must be a positive integer")
        max_pending = int(self.max_pending)
        if max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "model_dir", model_dir)
        object.__setattr__(self, "max_buffer_seconds", max_buffer_seconds)
        object.__setattr__(self, "update_interval", update_interval)
        object.__setattr__(self, "max_pending", max_pending)
        object.__setattr__(self, "max_audio_seconds", max_audio_seconds)

    @classmethod
    def coerce(
        cls,
        value: Any,
        *,
        aligner: Any = None,
        language: Optional[str] = None,
    ) -> Optional["AlignmentConfig"]:
        """Normalize public shorthand values used by engine APIs."""

        if value is None or value is False:
            return None
        if isinstance(value, cls):
            updates = {}
            if aligner is not None:
                updates["aligner"] = aligner
            if language is not None and value.language is None:
                updates["language"] = language
            return replace(value, **updates) if updates else value
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"off", "none"}:
                return None
            if token != "auto":
                raise ValueError("alignment source must be 'auto' or 'off'")
            data: dict[str, Any] = {"source": "auto"}
        elif value is True:
            data = {"source": "auto"}
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            raise TypeError(
                "alignment must be None, False, True, 'auto', 'off', a mapping, "
                "or AlignmentConfig"
            )
        if aligner is not None:
            data["aligner"] = aligner
        if language is not None:
            data.setdefault("language", language)
        return cls(**data)


@dataclass(frozen=True)
class AlignmentSnapshot:
    """Immutable view of all alignment data at one timeline revision."""

    text: str
    words: tuple[AlignedSpan, ...] = ()
    characters: tuple[AlignedSpan, ...] = ()
    status: str = "idle"
    final: bool = False
    revision: int = 0
    duration: float = 0.0
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        words = tuple(self.words)
        characters = tuple(self.characters)
        if any(not isinstance(item, AlignedSpan) for item in words + characters):
            raise TypeError("words and characters must contain AlignedSpan objects")
        status = _label(self.status, "status")
        if isinstance(self.revision, bool) or not isinstance(self.revision, Integral):
            raise TypeError("revision must be a non-negative integer")
        revision = int(self.revision)
        if revision < 0:
            raise ValueError("revision must be non-negative")
        duration = _finite_float(self.duration, "duration")
        errors = tuple(str(error) for error in self.errors)
        object.__setattr__(self, "words", words)
        object.__setattr__(self, "characters", characters)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "errors", errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "words": [span.to_dict(alias="word") for span in self.words],
            "characters": [span.to_dict(alias="character") for span in self.characters],
            "status": self.status,
            "final": self.final,
            "revision": self.revision,
            "duration": self.duration,
            "errors": list(self.errors),
        }

    def export(self, path: Union[str, Path], format: Optional[str] = None) -> Path:
        """Write this snapshot as JSON, SRT, or WebVTT using UTF-8."""

        destination = Path(path)
        kind = (format or destination.suffix.lstrip(".") or "json").strip().lower()
        kind = {"webvtt": "vtt"}.get(kind, kind)
        if kind == "json":
            content = json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
        elif kind == "srt":
            content = _render_subtitles(self, webvtt=False)
        elif kind == "vtt":
            content = _render_subtitles(self, webvtt=True)
        else:
            raise ValueError("format must be one of 'json', 'srt', or 'vtt'")
        with destination.open("w", encoding="utf-8", newline="\n") as output:
            output.write(content)
        return destination


def _format_timestamp(seconds: float, *, webvtt: bool) -> str:
    milliseconds = int(round(max(0.0, seconds) * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_value, milliseconds = divmod(remainder, 1_000)
    separator = "." if webvtt else ","
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d}{separator}{milliseconds:03d}"


def _cue_text(snapshot: AlignmentSnapshot, spans: list[AlignedSpan]) -> str:
    first, last = spans[0], spans[-1]
    if (
        first.text_start is not None
        and last.text_end is not None
        and 0 <= first.text_start <= last.text_end <= len(snapshot.text)
    ):
        source = snapshot.text[first.text_start:last.text_end].strip()
        if source:
            return source
    return " ".join(span.text.strip() for span in spans if span.text.strip()).strip()


def _subtitle_cues(snapshot: AlignmentSnapshot) -> list[tuple[float, float, str]]:
    spans = list(snapshot.words)
    if spans:
        cues: list[tuple[float, float, str]] = []
        current: list[AlignedSpan] = []
        for span in spans:
            if current:
                gap = span.start_time - current[-1].end_time
                if gap > 0.75 or _SENTENCE_END.search(current[-1].text.strip()) or len(current) >= 8:
                    cues.append((current[0].start_time, max(item.end_time for item in current), _cue_text(snapshot, current)))
                    current = []
            current.append(span)
        if current:
            cues.append((current[0].start_time, max(item.end_time for item in current), _cue_text(snapshot, current)))
        return cues
    if snapshot.characters:
        chars = list(snapshot.characters)
        text = _cue_text(snapshot, chars)
        return [(chars[0].start_time, chars[-1].end_time, text)] if text else []
    if snapshot.text.strip():
        return [(0.0, snapshot.duration, snapshot.text.strip())]
    return []


def _render_subtitles(snapshot: AlignmentSnapshot, *, webvtt: bool) -> str:
    cues = _subtitle_cues(snapshot)
    rendered: list[str] = ["WEBVTT", ""] if webvtt else []
    for index, (start, end, text) in enumerate(cues, 1):
        end = max(end, start + 0.001)
        if webvtt:
            safe_text = (
                text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\r\n", "\n").replace("\r", "\n")
            )
            rendered.extend([
                str(index),
                f"{_format_timestamp(start, webvtt=True)} --> {_format_timestamp(end, webvtt=True)}",
                safe_text,
                "",
            ])
        else:
            rendered.extend([
                str(index),
                f"{_format_timestamp(start, webvtt=False)} --> {_format_timestamp(end, webvtt=False)}",
                text.replace("\r\n", "\n").replace("\r", "\n"),
                "",
            ])
    return "\n".join(rendered).rstrip("\n") + "\n"


class AlignmentTimeline:
    """Thread-safe mutable timeline with immutable callback snapshots."""

    def __init__(
        self,
        text: str = "",
        *,
        on_update: Optional[Callable[[AlignmentSnapshot], None]] = None,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        self._condition = threading.Condition(threading.RLock())
        self._on_update = on_update
        self._text = text
        self._fragments: dict[str, tuple[AlignedSpan, ...]] = {}
        self._status = "idle"
        self._final = False
        self._revision = 0
        self._duration = 0.0
        self._position = 0.0
        self._errors: list[str] = []

    def _ordered_spans_locked(self) -> tuple[AlignedSpan, ...]:
        entries: list[tuple[int, AlignedSpan]] = []
        serial = 0
        for spans in self._fragments.values():
            for span in spans:
                entries.append((serial, span))
                serial += 1
        entries.sort(
            key=lambda item: (
                item[1].start_time,
                item[1].end_time,
                item[1].text_start if item[1].text_start is not None else 10**18,
                item[0],
            )
        )
        return tuple(item[1] for item in entries)

    def _snapshot_locked(self) -> AlignmentSnapshot:
        spans = self._ordered_spans_locked()
        return AlignmentSnapshot(
            text=self._text,
            words=tuple(span for span in spans if span.level == "word"),
            characters=tuple(span for span in spans if span.level == "character"),
            status=self._status,
            final=self._final,
            revision=self._revision,
            duration=self._duration,
            errors=tuple(self._errors),
        )

    def _emit(self, snapshot: AlignmentSnapshot) -> None:
        callback = self._on_update
        if callback is None:
            return
        try:
            callback(snapshot)
        except Exception:
            logging.exception("Alignment timeline update callback failed")

    @staticmethod
    def _fragment_id(fragment_id: Any) -> str:
        if fragment_id is None:
            return ""
        if not isinstance(fragment_id, str):
            raise TypeError("fragment_id must be a string")
        return fragment_id

    @staticmethod
    def _coerce_span(value: Any, fragment_id: str) -> AlignedSpan:
        if isinstance(value, AlignedSpan):
            span = value
        elif isinstance(value, Mapping):
            payload = dict(value)
            if "text" not in payload:
                if "word" in payload:
                    payload["text"] = payload.pop("word")
                elif "character" in payload:
                    payload["text"] = payload.pop("character")
                else:
                    raise ValueError("alignment mapping requires text, word, or character")
            payload.pop("index", None)
            span = AlignedSpan(**payload)
        else:
            raise TypeError("publish() accepts AlignedSpan objects or mappings")
        if span.fragment_id != fragment_id:
            span = replace(span, fragment_id=fragment_id)
        return span

    def reset(self, text: str = "") -> AlignmentSnapshot:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        with self._condition:
            self._text = text
            self._fragments.clear()
            self._status = "idle"
            self._final = False
            self._duration = 0.0
            self._position = 0.0
            self._errors.clear()
            self._revision += 1
            snapshot = self._snapshot_locked()
            self._condition.notify_all()
        self._emit(snapshot)
        return snapshot

    def set_text(self, text: str) -> AlignmentSnapshot:
        """Update the source text without clearing published alignment data."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        with self._condition:
            # A final or cancelled timeline is fenced until the caller starts
            # a new generation with reset(); late streamed text cannot mutate
            # the completed snapshot.
            if self._final or text == self._text:
                return self._snapshot_locked()
            self._text = text
            self._revision += 1
            snapshot = self._snapshot_locked()
            self._condition.notify_all()
        self._emit(snapshot)
        return snapshot

    def publish(
        self,
        spans: Iterable[Union[AlignedSpan, Mapping[str, Any]]] | AlignedSpan,
        *,
        fragment_id: str,
        status: str = "aligning",
    ) -> AlignmentSnapshot:
        fragment_id = self._fragment_id(fragment_id)
        status = _label(status, "status")
        if isinstance(spans, AlignedSpan) or isinstance(spans, Mapping):
            values = (spans,)
        else:
            values = tuple(spans)
        replacement = tuple(self._coerce_span(value, fragment_id) for value in values)
        with self._condition:
            if self._final:
                return self._snapshot_locked()
            self._fragments[fragment_id] = replacement
            self._status = status
            self._revision += 1
            snapshot = self._snapshot_locked()
            self._condition.notify_all()
        self._emit(snapshot)
        return snapshot

    def set_duration(self, seconds: float) -> AlignmentSnapshot:
        duration = _finite_float(seconds, "duration")
        with self._condition:
            if self._final:
                return self._snapshot_locked()
            self._duration = duration
            self._position = min(self._position, duration)
            self._revision += 1
            snapshot = self._snapshot_locked()
            self._condition.notify_all()
        self._emit(snapshot)
        return snapshot

    def set_status(self, status: str) -> AlignmentSnapshot:
        """Update a nonterminal status while retaining all published data."""

        status = _label(status, "status")
        with self._condition:
            if self._final:
                return self._snapshot_locked()
            self._status = status
            self._revision += 1
            snapshot = self._snapshot_locked()
            self._condition.notify_all()
        self._emit(snapshot)
        return snapshot

    def finish(self, status: str = "ready") -> AlignmentSnapshot:
        status = _label(status, "status")
        with self._condition:
            if self._final:
                return self._snapshot_locked()
            self._status = status
            self._final = True
            self._revision += 1
            snapshot = self._snapshot_locked()
            self._condition.notify_all()
        self._emit(snapshot)
        return snapshot

    def cancel(self) -> AlignmentSnapshot:
        with self._condition:
            if self._final:
                return self._snapshot_locked()
            self._status = "cancelled"
            self._final = True
            self._revision += 1
            snapshot = self._snapshot_locked()
            self._condition.notify_all()
        self._emit(snapshot)
        return snapshot

    def error(self, message: str) -> AlignmentSnapshot:
        message = _label(message, "message")
        with self._condition:
            if self._final:
                return self._snapshot_locked()
            self._errors.append(message)
            self._status = "error"
            self._revision += 1
            snapshot = self._snapshot_locked()
            self._condition.notify_all()
        self._emit(snapshot)
        return snapshot

    def snapshot(self) -> AlignmentSnapshot:
        with self._condition:
            return self._snapshot_locked()

    def wait(self, timeout: Optional[float] = None) -> AlignmentSnapshot:
        if timeout is not None:
            timeout = _finite_float(timeout, "timeout")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._final:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("alignment did not reach a final state")
                self._condition.wait(remaining)
            return self._snapshot_locked()

    def advance(self, seconds: float) -> None:
        delta = _finite_float(seconds, "seconds")
        with self._condition:
            self._position += delta
            if self._duration > 0:
                self._position = min(self._position, self._duration)

    def seek(self, seconds: float) -> None:
        """Set the absolute output position, including a seek during replay."""
        position = _finite_float(seconds, "seconds")
        with self._condition:
            self._position = position

    def at(self, seconds: float, level: str = "word") -> Optional[AlignedSpan]:
        timestamp = _finite_float(seconds, "seconds")
        level = _level(level)
        with self._condition:
            candidates = [span for span in self._ordered_spans_locked() if span.level == level]
        for index, span in enumerate(candidates):
            if span.start_time <= timestamp < span.end_time:
                return span
            if index == len(candidates) - 1 and span.start_time == span.end_time == timestamp:
                return span
        return None

    @property
    def words(self) -> tuple[AlignedSpan, ...]:
        with self._condition:
            return tuple(span for span in self._ordered_spans_locked() if span.level == "word")

    @property
    def characters(self) -> tuple[AlignedSpan, ...]:
        with self._condition:
            return tuple(span for span in self._ordered_spans_locked() if span.level == "character")

    @property
    def current_word(self) -> Optional[AlignedSpan]:
        with self._condition:
            position = self._position
        return self.at(position, "word")

    @property
    def status(self) -> str:
        with self._condition:
            return self._status

    def export(self, path: Union[str, Path], format: Optional[str] = None) -> Path:
        return self.snapshot().export(path, format=format)


from .text_mapping import (
    map_alignment_offsets,
    map_text_offsets,
    project_text_offsets,
)
