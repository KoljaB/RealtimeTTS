"""Optional pitch-preserving tempo processing for native PCM chunks."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import threading
from typing import Any, Iterator

from . import _audio_formats as formats


_MIN_SPEED = 0.25
_MAX_SPEED = 4.0
_PCM_INFO = {
    formats.INT16: ("s16", "<i2", 2),
    formats.FLOAT32: ("flt", "<f4", 4),
}


def validate_speed(speed: float) -> float:
    """Validate and normalize one segment's tempo factor."""
    if isinstance(speed, bool):
        raise ValueError("speed must be a finite float between 0.25 and 4.0")
    try:
        value = float(speed)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "speed must be a finite float between 0.25 and 4.0"
        ) from exc
    if not math.isfinite(value) or not _MIN_SPEED <= value <= _MAX_SPEED:
        raise ValueError("speed must be a finite float between 0.25 and 4.0")
    return value


@dataclass(frozen=True, slots=True)
class SpeedChange:
    """Ordered marker for a new speed segment."""

    speed: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "speed", validate_speed(self.speed))


class TempoProcessor:
    """Stateful pitch-preserving tempo processor for interleaved PCM.

    A processor owns one immutable speed. Feed arbitrary byte chunks, consume
    whatever output is available, then call finish before starting a new
    segment. PyAV is imported only when speed differs from one.
    """

    def __init__(
        self,
        stream_format: int,
        channels: int,
        sample_rate: int,
        speed: float,
    ) -> None:
        try:
            self.stream_format = int(stream_format)
            self.channels = int(channels)
            self.sample_rate = int(sample_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "stream format, channels, and sample rate must be integers"
            ) from exc
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        try:
            self._av_format, self._dtype, self._sample_width = _PCM_INFO[
                self.stream_format
            ]
        except KeyError as exc:
            raise ValueError(
                "TempoProcessor supports only packed PCM16 and float32 input"
            ) from exc

        self.speed = validate_speed(speed)
        self._bypass = self.speed == 1.0
        self._frame_bytes = self.channels * self._sample_width
        self._processing_bytes = self._frame_bytes * 256
        self._minimum_input_frames = max(
            1024, math.ceil(self.sample_rate * 1024 / 24_000)
        )
        self._layout = self._layout_for_channels(self.channels)
        self._pending = bytearray()
        self._frame_buffer = bytearray()
        self._finished = False
        self._input_frames = 0
        self._input_frames_received = 0
        self._av: Any = None
        self._np: Any = None
        self._graph: Any = None
        self._output_resampler: Any = None
        self._drain_errors: tuple[type[BaseException], ...] = ()

        if not self._bypass:
            self._build_backend()

    @staticmethod
    def _layout_for_channels(channels: int) -> str:
        if channels == 1:
            return "mono"
        if channels == 2:
            return "stereo"
        return f"{channels}c"

    @staticmethod
    def _tempo_factors(speed: float) -> tuple[float, ...]:
        remaining = speed
        factors: list[float] = []
        while remaining > 2.0:
            factors.append(2.0)
            remaining /= 2.0
        while remaining < 0.5:
            factors.append(0.5)
            remaining /= 0.5
        factors.append(remaining)
        return tuple(factors)

    def _build_backend(self) -> None:
        try:
            import av
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "TempoProcessor with speed != 1 requires PyAV; "
                "install realtimetts[speed]"
            ) from exc
        try:
            av.AudioLayout(self._layout)
        except Exception as exc:
            raise ValueError(
                f"unsupported channel count for PyAV audio layout: {self.channels}"
            ) from exc

        graph = av.filter.Graph()
        source = graph.add_abuffer(
            sample_rate=self.sample_rate,
            format=self._av_format,
            layout=self._layout,
            time_base=Fraction(1, self.sample_rate),
        )
        nodes = [source]
        nodes.extend(
            graph.add("atempo", str(factor))
            for factor in self._tempo_factors(self.speed)
        )
        sink = graph.add("abuffersink")
        nodes.append(sink)
        graph.link_nodes(*nodes).configure()

        self._av = av
        self._np = np
        self._graph = graph
        self._output_resampler = av.AudioResampler(
            format=self._av_format,
            layout=self._layout,
            rate=self.sample_rate,
        )
        self._drain_errors = (av.BlockingIOError, av.EOFError)

    def _discard_backend(self) -> None:
        self._graph = None
        self._output_resampler = None

    def _ensure_active(self) -> None:
        if self._finished:
            raise RuntimeError("TempoProcessor has already been finished")

    def _frame_from_bytes(self, payload: bytes) -> Any:
        values = self._np.frombuffer(payload, dtype=self._dtype)
        if values.size % self.channels:
            raise ValueError("PCM payload does not contain whole audio frames")
        frame = self._av.AudioFrame.from_ndarray(
            values.reshape(1, -1),
            format=self._av_format,
            layout=self._layout,
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self._input_frames
        frame.time_base = Fraction(1, self.sample_rate)
        self._input_frames += values.size // self.channels
        return frame

    def _frame_to_bytes(self, frame: Any) -> bytes:
        values = self._np.asarray(frame.to_ndarray()).reshape(-1)
        values = self._np.ascontiguousarray(values.astype(self._dtype, copy=False))
        payload = values.tobytes()
        if len(payload) % self._frame_bytes:
            raise RuntimeError("PyAV returned a partial PCM audio frame")
        return payload

    def _drain_graph(self) -> list[bytes]:
        output: list[bytes] = []
        while True:
            try:
                frame = self._graph.pull()
            except self._drain_errors:
                break
            for converted in self._output_resampler.resample(frame):
                payload = self._frame_to_bytes(converted)
                if payload:
                    output.append(payload)
        return output

    def _process_complete(self, payload: bytes) -> list[bytes]:
        self._graph.push(self._frame_from_bytes(payload))
        return self._drain_graph()

    def feed(self, data: bytes | bytearray | memoryview) -> Iterator[bytes]:
        """Feed bytes and return currently available output chunks."""
        self._ensure_active()
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("feed expects a bytes-like PCM payload")
        incoming = bytes(data)
        if not incoming:
            return iter(())
        combined = bytes(self._pending) + incoming
        usable = len(combined) - (len(combined) % self._frame_bytes)
        if usable == 0:
            self._pending = bytearray(combined)
            return iter(())
        payload = combined[:usable]
        self._pending = bytearray(combined[usable:])
        if self._bypass:
            return iter((payload,))

        self._frame_buffer.extend(payload)
        self._input_frames_received += len(payload) // self._frame_bytes
        if self._input_frames_received < self._minimum_input_frames:
            return iter(())
        output: list[bytes] = []
        while len(self._frame_buffer) >= self._processing_bytes:
            block = bytes(self._frame_buffer[:self._processing_bytes])
            del self._frame_buffer[:self._processing_bytes]
            output.extend(self._process_complete(block))
        return iter(output)

    def finish(self) -> Iterator[bytes]:
        """Flush the graph and return all final output chunks."""
        if self._finished:
            return iter(())
        if self._pending:
            pending = len(self._pending)
            self._pending.clear()
            self._frame_buffer.clear()
            self._finished = True
            self._discard_backend()
            raise ValueError(
                f"finish received {pending} trailing bytes; "
                f"PCM frame size is {self._frame_bytes}"
            )
        self._finished = True
        if self._bypass:
            return iter(())
        output: list[bytes] = []
        short_segment = self._input_frames_received < self._minimum_input_frames
        target_bytes = 0
        if short_segment and self._frame_buffer:
            target_frames = max(
                1, round(self._input_frames_received / self.speed)
            )
            target_bytes = target_frames * self._frame_bytes
            last_frame = bytes(self._frame_buffer[-self._frame_bytes:])
            self._frame_buffer.extend(
                last_frame
                * (self._minimum_input_frames - self._input_frames_received)
            )
        try:
            while len(self._frame_buffer) >= self._processing_bytes:
                block = bytes(self._frame_buffer[:self._processing_bytes])
                del self._frame_buffer[:self._processing_bytes]
                output.extend(self._process_complete(block))
            if self._frame_buffer:
                output.extend(self._process_complete(bytes(self._frame_buffer)))
                self._frame_buffer.clear()
            self._graph.push(None)
            output.extend(self._drain_graph())
            for converted in self._output_resampler.resample(None):
                payload = self._frame_to_bytes(converted)
                if payload:
                    output.append(payload)
        finally:
            self._discard_backend()
        if short_segment:
            trimmed = b"".join(output)[:target_bytes]
            output = [trimmed] if trimmed else []
        return iter(output)

    def reset(self) -> None:
        """Discard buffered audio and make the processor usable again."""
        self._pending.clear()
        self._frame_buffer.clear()
        self._input_frames = 0
        self._input_frames_received = 0
        self._finished = False
        if not self._bypass:
            self._discard_backend()
            self._build_backend()

    def cancel(self) -> None:
        """Discard the current segment without rebuilding the filter graph."""
        self._pending.clear()
        self._frame_buffer.clear()
        self._input_frames = 0
        self._input_frames_received = 0
        self._finished = True
        self._discard_backend()


class DecodedAudioQueue:
    """Decode compressed MP3 chunks before forwarding float32 mono PCM."""

    def __init__(self, wrapped: Any, sample_rate: int = 24_000) -> None:
        self.wrapped = wrapped
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.output_format = (formats.FLOAT32, 1, self.sample_rate)
        self._lock = threading.Lock()
        self._active = False
        self.begin()

    def begin(self) -> None:
        with self._lock:
            if "_decoder" in self.__dict__:
                self._decoder.reset()
            else:
                from .alignment.decoder import MP3Decoder

                self._decoder = MP3Decoder(self.sample_rate)
            self._active = True

    def put(self, item: Any, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            if not self._active:
                return
            for chunk in self._decoder.feed(bytes(item)):
                if not self._active:
                    return
                self.wrapped.put(chunk, *args, **kwargs)

    def put_nowait(self, item: Any) -> None:
        self.put(item, block=False)

    def finish(self) -> None:
        with self._lock:
            if not self._active:
                return
            for chunk in self._decoder.finish():
                if not self._active:
                    return
                self.wrapped.put(chunk)
            self._active = False

    def cancel(self) -> None:
        with self._lock:
            self._active = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)


__all__ = ["DecodedAudioQueue", "SpeedChange", "TempoProcessor", "validate_speed"]
