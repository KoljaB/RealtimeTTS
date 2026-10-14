"""Thread-safe source/output timing for piecewise tempo changes."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple


_EPSILON = 1e-12
RETIMED_SOURCE = "retimed_derived"


def _nonnegative(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _positive(value: float, name: str = "speed") -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


@dataclass(frozen=True, slots=True)
class TempoSegment:
    """An immutable source/output clock segment.

    speed is the requested factor (>1 is faster). A closed segment maps using
    its actual observed endpoints. An open snapshot segment uses speed as its
    provisional slope and is therefore derived timing evidence.
    """

    source_start: float
    source_end: float
    output_start: float
    output_end: float
    speed: float
    closed: bool = True

    def __post_init__(self) -> None:
        source_start = _nonnegative(self.source_start, "source_start")
        source_end = _nonnegative(self.source_end, "source_end")
        output_start = _nonnegative(self.output_start, "output_start")
        output_end = _nonnegative(self.output_end, "output_end")
        speed = _positive(self.speed)
        if source_end + _EPSILON < source_start:
            raise ValueError("source_end must not precede source_start")
        if output_end + _EPSILON < output_start:
            raise ValueError("output_end must not precede output_start")
        object.__setattr__(self, "source_start", source_start)
        object.__setattr__(self, "source_end", source_end)
        object.__setattr__(self, "output_start", output_start)
        object.__setattr__(self, "output_end", output_end)
        object.__setattr__(self, "speed", speed)
        object.__setattr__(self, "closed", bool(self.closed))

    @property
    def source_duration(self) -> float:
        return self.source_end - self.source_start

    @property
    def output_duration(self) -> float:
        return self.output_end - self.output_start

    @property
    def timing_source(self) -> str:
        return RETIMED_SOURCE

    @property
    def quality(self) -> str:
        return RETIMED_SOURCE

    @property
    def derived(self) -> bool:
        return True


def _source_in_segment(segment: TempoSegment, value: float) -> float:
    if segment.source_duration <= _EPSILON:
        return segment.output_end
    if value <= segment.source_start:
        return segment.output_start
    if value >= segment.source_end:
        return segment.output_end
    if (
        segment.speed == 1.0
        and segment.source_start == segment.output_start
        and segment.source_duration == segment.output_duration
    ):
        return value
    return segment.output_start + (
        (value - segment.source_start)
        * segment.output_duration
        / segment.source_duration
    )


def _output_in_segment(segment: TempoSegment, value: float) -> float:
    if segment.output_duration <= _EPSILON:
        return segment.source_end
    if value <= segment.output_start:
        return segment.source_start
    if value >= segment.output_end:
        return segment.source_end
    if (
        segment.speed == 1.0
        and segment.source_start == segment.output_start
        and segment.source_duration == segment.output_duration
    ):
        return value
    return segment.source_start + (
        (value - segment.output_start)
        * segment.source_duration
        / segment.output_duration
    )


def _source_open(segment: TempoSegment, value: float) -> float:
    return segment.output_start + max(0.0, value - segment.source_start) / segment.speed


def _output_open(segment: TempoSegment, value: float) -> float:
    if value <= segment.output_start:
        return segment.source_start
    target = segment.source_start + (
        value - segment.output_start
    ) * segment.speed
    if value <= segment.output_end + _EPSILON:
        return min(target, segment.source_end)
    known = min(
        segment.source_start
        + (segment.output_end - segment.output_start) * segment.speed,
        segment.source_end,
    )
    return known + (value - segment.output_end) * segment.speed


def _map_source(segments: Tuple[TempoSegment, ...], value: float) -> float:
    value = max(0.0, float(value))
    boundary: Optional[TempoSegment] = None
    for segment in segments:
        if value + _EPSILON < segment.source_start:
            break
        if value <= segment.source_end + _EPSILON:
            boundary = segment
            if value < segment.source_end - _EPSILON:
                return (
                    _source_in_segment(segment, value)
                    if segment.closed else _source_open(segment, value)
                )
            if segment.source_duration > _EPSILON:
                return (
                    _source_in_segment(segment, value)
                    if segment.closed else _source_open(segment, value)
                )
        elif segment.source_end < value - _EPSILON:
            continue
    if boundary is not None:
        return (
            _source_in_segment(boundary, value)
            if boundary.closed else _source_open(boundary, value)
        )
    if not segments:
        return value
    last = segments[-1]
    if last.closed:
        return last.output_end + max(0.0, value - last.source_end) / last.speed
    return _source_open(last, value)


def _map_output(segments: Tuple[TempoSegment, ...], value: float) -> float:
    value = max(0.0, float(value))
    boundary: Optional[TempoSegment] = None
    for segment in segments:
        if value + _EPSILON < segment.output_start:
            break
        if value <= segment.output_end + _EPSILON:
            boundary = segment
            if value < segment.output_end - _EPSILON:
                return (
                    _output_in_segment(segment, value)
                    if segment.closed else _output_open(segment, value)
                )
            if segment.output_duration > _EPSILON:
                return (
                    _output_in_segment(segment, value)
                    if segment.closed else _output_open(segment, value)
                )
        elif segment.output_end < value - _EPSILON:
            continue
    if boundary is not None:
        return (
            _output_in_segment(boundary, value)
            if boundary.closed else _output_open(boundary, value)
        )
    if not segments:
        return value
    last = segments[-1]
    if last.closed:
        return last.source_end + max(0.0, value - last.output_end) * last.speed
    return _output_open(last, value)


class TempoTimeline:
    """Thread-safe piecewise mapping between source and retimed output time."""

    mapping_source = RETIMED_SOURCE

    def __init__(self, default_speed: float = 1.0):
        self._lock = threading.RLock()
        self._default_speed = _positive(default_speed, "default_speed")
        self._generation = 0
        self._closed_segments: list[TempoSegment] = []
        self._source_duration = 0.0
        self._output_duration = 0.0
        self._current_speed: Optional[float] = self._default_speed
        self._current_start_source = 0.0
        self._current_start_output = 0.0

    @property
    def default_speed(self) -> float:
        return self._default_speed

    @property
    def source_duration(self) -> float:
        with self._lock:
            return self._source_duration

    @property
    def output_duration(self) -> float:
        with self._lock:
            return self._output_duration

    @property
    def current_speed(self) -> Optional[float]:
        with self._lock:
            return self._current_speed

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def changed(self) -> bool:
        with self._lock:
            return (
                any(segment.speed != 1.0 for segment in self._closed_segments)
                or (
                    self._current_speed is not None
                    and self._current_speed != 1.0
                )
            )

    @property
    def segments(self) -> Tuple[TempoSegment, ...]:
        return self.snapshot()

    @property
    def snapshot_segments(self) -> Tuple[TempoSegment, ...]:
        return self.snapshot()

    def _open_locked(self, speed: float) -> None:
        self._current_speed = speed
        self._current_start_source = self._source_duration
        self._current_start_output = self._output_duration

    def _current_locked(self) -> Optional[TempoSegment]:
        if self._current_speed is None:
            return None
        return TempoSegment(
            self._current_start_source,
            self._source_duration,
            self._current_start_output,
            self._output_duration,
            self._current_speed,
            closed=False,
        )

    def _close_locked(self) -> Optional[TempoSegment]:
        segment = self._current_locked()
        closed = None
        if segment is not None and (
            segment.source_duration > _EPSILON
            or segment.output_duration > _EPSILON
        ):
            closed = TempoSegment(
                segment.source_start,
                segment.source_end,
                segment.output_start,
                segment.output_end,
                segment.speed,
                closed=True,
            )
            self._closed_segments.append(closed)
        self._current_speed = None
        return closed

    def begin(self, speed: float) -> Optional[TempoSegment]:
        """Close the current non-empty segment and start a speed."""
        with self._lock:
            previous = self._close_locked()
            self._open_locked(_positive(speed))
            return previous

    def add_input(self, seconds: float) -> float:
        """Record source duration fed to the tempo processor."""
        seconds = _nonnegative(seconds, "seconds")
        with self._lock:
            if self._current_speed is None:
                self._open_locked(self._default_speed)
            self._source_duration += seconds
            return self._source_duration

    def add_output(self, seconds: float) -> float:
        """Record output duration emitted by the tempo processor."""
        seconds = _nonnegative(seconds, "seconds")
        with self._lock:
            if self._current_speed is None:
                self._open_locked(self._default_speed)
            self._output_duration += seconds
            return self._output_duration

    def finish_segment(self, next_speed: Optional[float] = None) -> Optional[TempoSegment]:
        """Finalize the current segment and optionally open the next speed."""
        if next_speed is not None:
            next_speed = _positive(next_speed)
        with self._lock:
            segment = self._close_locked()
            if next_speed is not None:
                self._open_locked(next_speed)
            return segment

    def reset(self, speed: Optional[float] = None) -> None:
        """Discard history and start a fresh generation."""
        if speed is None:
            speed = self._default_speed
        speed = _positive(speed)
        with self._lock:
            self._generation += 1
            self._closed_segments.clear()
            self._source_duration = 0.0
            self._output_duration = 0.0
            self._open_locked(speed)

    def snapshot(self) -> Tuple[TempoSegment, ...]:
        """Return an immutable tuple of closed segments and the current one."""
        with self._lock:
            segments = list(self._closed_segments)
            current = self._current_locked()
            if current is not None:
                segments.append(current)
            return tuple(segments)

    def source_to_output(self, seconds: float) -> float:
        with self._lock:
            return _map_source(self.snapshot(), seconds)

    def output_to_source(self, seconds: float) -> float:
        with self._lock:
            return _map_output(self.snapshot(), seconds)


__all__ = ["RETIMED_SOURCE", "TempoSegment", "TempoTimeline"]
