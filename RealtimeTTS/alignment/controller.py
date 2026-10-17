"""Engine-independent alignment sessions for PCM and compressed TTS streams."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
import logging
import queue
import re
import threading
import time
import unicodedata

import numpy as np

from .. import _audio_formats as formats
from ..audio_tempo import DecodedAudioQueue
from .audio import audio_bytes_to_mono_float32, pcm_chunk_duration_seconds
from .backends import CPUAlignmentModel, align_snapshot, make_alignment_model
from .evidence import coverage_counts
from .timeline import AlignedSpan, AlignmentConfig, AlignmentTimeline

LOGGER = logging.getLogger(__name__)


def _normalized_map(text):
    characters, ranges = [], []
    for index, character in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", character).casefold():
            normalized = normalized if normalized.isalnum() or normalized == "'" else " "
            if normalized == " " and (not characters or characters[-1] == " "):
                continue
            characters.append(normalized)
            ranges.append((index, index + 1))
    return "".join(characters), ranges


def _ranges_from_utf16(text, ranges):
    boundaries, offset = {0: 0}, 0
    for index, character in enumerate(text):
        offset += len(character.encode("utf-16-le")) // 2
        boundaries[offset] = index + 1
    return [(boundaries.get(start), boundaries.get(end)) for start, end in ranges]


@dataclass
class _Fragment:
    id: str
    generation: int
    text: str
    original: str
    text_offset: int | None
    start: float
    rate: int
    engine: object
    engine_start: float
    native: bool
    ranges: list | None = None
    samples: int = 0
    covered: int = 0
    expected: int = 0
    chunks: list = field(default_factory=list)
    cursor: object | None = None
    audio_start: int = 0
    items: dict = field(default_factory=lambda: {"words": [], "characters": []})
    finished: bool = False
    unavailable: bool = False
    native_done: bool = False
    native_unverified: bool = False
    scheduled_samples: int = 0
    version: int = 0
    done: threading.Event = field(default_factory=threading.Event)
    created: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    native_session: object | None = None


class _QueueProxy:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def put_nowait(self, item):
        return self.put(item, block=False)


class _AudioQueue(_QueueProxy):
    def __init__(self, wrapped, binding):
        super().__init__(wrapped)
        self.binding = binding

    def put(self, item, *args, **kwargs):
        binding = self.binding
        if binding.decoder is not None:
            for decoded in binding.decoder.feed(item):
                binding.publish(decoded, *args, **kwargs)
            return None
        return binding.publish(item, *args, **kwargs)


class _TimingQueue(_QueueProxy):
    def __init__(self, wrapped, binding):
        super().__init__(wrapped)
        self.binding = binding

    def put(self, item, *args, **kwargs):
        self.binding.owner.native_timing(self.binding.engine, item)
        return self.wrapped.put(item, *args, **kwargs)


class EngineAlignmentBinding:
    """One detachable queue adapter; the synthesis engine remains unchanged."""

    def __init__(self, owner, engine):
        self.owner, self.engine = owner, engine
        self._restore_queue = engine.queue
        self.original_queue = engine.queue
        if isinstance(self.original_queue, DecodedAudioQueue):
            # Alignment owns decoding now; never send PCM into a prior MP3 decoder.
            self.original_queue.cancel()
            self.original_queue = self.original_queue.wrapped
        self.original_timings = engine.timings
        self.original_callback = getattr(engine, "on_alignment", None)
        self.input_format = tuple(engine.get_stream_info())
        self.output_format = self.input_format
        self.decoder = None
        if self.input_format[0] == formats.CUSTOM:
            from .decoder import MP3Decoder
            rate = self.input_format[2] if self.input_format[2] > 0 else 24000
            self.decoder = MP3Decoder(rate)
            self.output_format = (formats.FLOAT32, 1, rate)
        self.audio_queue = _AudioQueue(self.original_queue, self)
        self.timing_queue = _TimingQueue(self.original_timings, self)
        engine.queue, engine.timings = self.audio_queue, self.timing_queue
        self.fragment = None

    def begin(self, text, *, original_text=None, source_ranges=None, dynamic=False):
        if self.decoder:
            self.decoder.reset()
        self.fragment = self.owner.begin_fragment(
            self.engine, text, self.output_format, original_text=original_text,
            source_ranges=source_ranges, dynamic=dynamic,
        )
        fragment = self.fragment
        self.previous_session = getattr(self.engine, "last_alignment", None)
        if hasattr(self.engine, "on_alignment"):
            previous = self.original_callback

            def aligned(event):
                self.owner.engine_event(fragment, event)
                if previous is not None:
                    previous(event)

            self.engine.on_alignment = aligned
        return fragment

    def append_text(self, text):
        if self.fragment is not None:
            self.owner.append_fragment_text(self.fragment, text)

    def publish(self, audio, *args, **kwargs):
        # Copy bytes for the analysis tap; never delay PCM on model inference.
        payload = bytes(audio)
        if self.fragment is not None and self.fragment.generation != self.owner._generation:
            return None
        self.owner.audio(self.fragment, payload, self.output_format)
        return self.original_queue.put(payload, *args, **kwargs)

    def finish(self, *, success=True):
        fragment = self.fragment
        try:
            if self.decoder is not None:
                for decoded in self.decoder.finish():
                    self.publish(decoded)
        finally:
            if hasattr(self.engine, "on_alignment"):
                self.engine.on_alignment = self.original_callback
            session = getattr(self.engine, "last_alignment", None)
            asynchronous = session is not None and session is not self.previous_session
            if fragment is not None and asynchronous:
                fragment.native_session = session
            self.owner.end_fragment(fragment, success=success, asynchronous=asynchronous)
            self.fragment = None

    def silence(self, seconds):
        fmt, channels, rate = self.output_format
        widths = {formats.FLOAT32: 4, formats.INT16: 2, formats.INT32: 4,
                  formats.INT8: 1, formats.UINT8: 1}
        frames = max(0, round(seconds * rate))
        fill = b"\x80" if fmt == formats.UINT8 else b"\0"
        payload = fill * (frames * channels * widths[fmt])
        self.owner.audio(None, payload, self.output_format)
        self.original_queue.put(payload)

    def detach(self):
        if self.engine.queue is self.audio_queue:
            self.engine.queue = self._restore_queue
        if self.engine.timings is self.timing_queue:
            self.engine.timings = self.original_timings
        if hasattr(self.engine, "on_alignment"):
            self.engine.on_alignment = self.original_callback


class AlignmentController:
    """Public timeline plus a bounded background alignment worker.

    Audio timestamps are relative to the complete output stream. Text offsets
    use Unicode codepoints in the exact input text.
    """

    def __init__(self, config, *, on_update=None, on_word=None, on_character=None):
        self.config = AlignmentConfig.coerce(config)
        if self.config is None:
            raise ValueError("An enabled alignment configuration is required")
        self._on_update, self._on_word, self._on_character = on_update, on_word, on_character
        self.timeline = AlignmentTimeline(on_update=self._updated)
        self._condition = threading.Condition(threading.RLock())
        self._generation = 0
        self._fragments = OrderedDict()
        self._pending = OrderedDict()
        self._binding = None
        self._model = None
        self._model_error = None
        self._closed = False
        self._finishing = False
        self._duration = 0.0
        self._source = ""
        self._source_cursor = 0
        self._position = 0.0
        self._playback_active = False
        self._fired = set()
        self._sync_waited = set()
        self._worker = threading.Thread(target=self._run, name="tts-shared-alignment", daemon=True)
        self._notification = threading.Event()
        self._latest_update = None
        self._notifier = threading.Thread(target=self._notify, name="tts-alignment-events", daemon=True)
        self._notifier.start()
        self._worker.start()

    def __getattr__(self, name):
        return getattr(self.timeline, name)

    def bind(self, engine):
        if self._binding is not None:
            self._binding.detach()
        self._binding = EngineAlignmentBinding(self, engine)
        return self._binding

    def reset(self):
        with self._condition:
            self._generation += 1
            for fragment in self._fragments.values():
                fragment.done.set()
            self._fragments.clear()
            self._pending.clear()
            self._duration = self._position = 0.0
            self._source, self._source_cursor = "", 0
            self._fired.clear()
            self._sync_waited.clear()
            self._playback_active = False
            self._finishing = False
            self._model_error = None
            self.timeline.reset()
            self._condition.notify_all()

    def append_source(self, text):
        with self._condition:
            self._source += text
            source = self._source
        self.timeline.set_text(source)

    def prepare(self):
        """Load a configured fallback model before requesting first speech."""
        with self._condition:
            if self._model is None:
                engine = self._binding.engine if self._binding else None
                self._model = make_alignment_model(self.config, engine)
        return self

    def _locate(self, text):
        # Prefer the actual substring: leading punctuation/emoji belong to the
        # fragment too and must not be counted again in relative token offsets.
        exact = self._source.find(text, self._source_cursor) if text else -1
        if exact >= 0:
            return exact
        source, ranges = _normalized_map(self._source)
        fragment, fragment_ranges = _normalized_map(text)
        fragment = fragment.strip()
        start = next((i for i, span in enumerate(ranges) if span[0] >= self._source_cursor), len(ranges))
        match = source.find(fragment, start) if fragment else -1
        if match < 0:
            return None
        candidate = ranges[match][0] - fragment_ranges[0][0]
        if candidate < self._source_cursor:
            return None
        # A length-changing rewrite needs an explicit source map. Do not
        # fabricate exact offsets from a normalized match alone.
        observed, _ = _normalized_map(self._source[candidate:candidate + len(text)])
        return candidate if observed.strip() == fragment else None

    def begin_fragment(self, engine, text, stream_format, *, original_text=None,
                       source_ranges=None, dynamic=False):
        with self._condition:
            original = text if original_text is None else original_text
            if not self._source and original:
                self.append_source(original)
            offset = self._locate(original)
            if dynamic:
                offset = self._source_cursor
            provides = bool(getattr(engine, "provides_word_timings", False))
            character_provider = "character" in getattr(
                getattr(engine, "streaming_aligner", None), "timestamp_levels", ()
            )
            native = self.config.source != "model" and provides and (
                self.config.level == "word" or character_provider
            )
            mapped_ranges = _ranges_from_utf16(original, source_ranges) if source_ranges else None
            ignored = getattr(engine, "alignment_non_speech_ranges", None)
            ignored = ignored(text) if callable(ignored) else ()
            if ignored:
                excluded = _ranges_from_utf16(text, [(span["start"], span["end"]) for span in ignored])
                indices = [i for i in range(len(text))
                           if not any(a is not None and b is not None and a <= i < b for a, b in excluded)]
                mapped_ranges = [(mapped_ranges[i] if mapped_ranges else (i, i + 1)) for i in indices]
                text = "".join(text[i] for i in indices)
            identifier = f"{self._generation}:{len(self._fragments) + 1}"
            fragment = _Fragment(
                identifier, self._generation, text, original, offset,
                self._duration, stream_format[2], engine,
                float(getattr(engine, "audio_duration", 0.0)), native,
                mapped_ranges,
            )
            fragment.expected = coverage_counts(text, ())[1]
            self._fragments[identifier] = fragment
            if self.config.source == "native" and not native:
                self._fail(fragment, "Engine does not provide the requested native timestamp level")
            return fragment

    def append_fragment_text(self, fragment, text):
        with self._condition:
            if fragment.generation != self._generation:
                return
            fragment.text += text
            fragment.original += text
            fragment.expected = coverage_counts(fragment.text, ())[1]
            fragment.version += 1

    def audio(self, fragment, payload, stream_format):
        duration = pcm_chunk_duration_seconds(payload, *stream_format)
        if duration is None:
            raise ValueError("Alignment output requires a decoded PCM stream")
        with self._condition:
            self._duration += duration
            self.timeline.set_duration(self._duration)
            if fragment is None or fragment.generation != self._generation or fragment.unavailable:
                return
            fragment.samples += round(duration * fragment.rate)
            if fragment.native:
                return
            if fragment.samples / fragment.rate > self.config.max_audio_seconds:
                self._fail(fragment, "Alignment audio limit exceeded; speech continues")
                return
            audio, _ = audio_bytes_to_mono_float32(payload, *stream_format)
            fragment.chunks.append(audio)
            fragment.version += 1
            partial = self.config.model_dir is not None or bool(
                getattr(self.config.aligner, "supports_partial", False)
                or hasattr(self.config.aligner, "backend")
            )
            if partial and self.config.mode != "export" and (
                fragment.samples - fragment.scheduled_samples >=
                self.config.update_interval * fragment.rate
            ):
                fragment.scheduled_samples = fragment.samples
                self._schedule(fragment)

    def _schedule(self, fragment):
        if fragment.id not in self._pending and len(self._pending) >= self.config.max_pending:
            _, dropped = self._pending.popitem(last=False)
            self._fail(dropped, "Alignment queue limit exceeded; speech continues")
        self._pending[fragment.id] = fragment
        self._condition.notify_all()

    def _fail(self, fragment, message):
        if fragment.unavailable:
            return
        fragment.unavailable = True
        fragment.chunks.clear()
        fragment.done.set()
        self._pending.pop(fragment.id, None)
        self.timeline.error(f"{fragment.id}: {message}")

    def end_fragment(self, fragment, *, success=True, asynchronous=False):
        if fragment is None:
            return
        with self._condition:
            if fragment.generation != self._generation:
                return
            fragment.finished = True
            fragment.finished_at = time.monotonic()
            if success and fragment.text_offset is not None:
                self._source_cursor = fragment.text_offset + len(fragment.original)
            if not success:
                self._fail(fragment, "Synthesis failed or was cancelled")
            elif fragment.native:
                if not asynchronous:
                    fragment.done.set()
                    self._check_native_result(fragment)
                self._publish(fragment)
            elif not fragment.unavailable:
                self._schedule(fragment)
            self._condition.notify_all()

    def _check_native_result(self, fragment):
        key = "characters" if self.config.level == "character" else "words"
        if fragment.text.strip() and not fragment.items[key] and not fragment.unavailable:
            self._fail(fragment, "Engine returned no timestamps for the requested level")

    def native_timing(self, engine, timing):
        # Embedded aligners emit fragment-bound events, which remain attributable
        # even when their worker finishes after the engine queue was restored.
        if getattr(engine, "streaming_aligner", None) is not None:
            return
        with self._condition:
            candidates = [f for f in self._fragments.values()
                          if f.engine is engine and f.native and not f.unavailable
                          and f.engine_start <= float(timing.start_time) + 1e-6]
            if not candidates:
                return
            fragment = candidates[-1]
            # Native times use the engine's clock; it excludes externally added pauses.
            item = {"word": timing.word,
                    "start_time": float(timing.start_time) - fragment.engine_start,
                    "end_time": float(timing.end_time) - fragment.engine_start}
            fragment.items["words"].append(item)
            self._publish(fragment)

    def engine_event(self, fragment, event):
        with self._condition:
            if fragment.generation != self._generation or fragment.unavailable or not fragment.native:
                return
            if event.get("type") == "alignment_error" or event.get("error"):
                self._fail(fragment, str(event.get("error", "Engine alignment failed")))
            delta = float(event.get("engine_audio_offset", fragment.engine_start)) - fragment.engine_start
            changed = False
            for key in ("words", "characters"):
                for item in event.get(key, ()):
                    fragment.items[key].append({
                        **item, "start_time": float(item["start_time"]) + delta,
                        "end_time": float(item["end_time"]) + delta,
                    })
                    changed = True
            if changed:
                self._publish(fragment)
            if event.get("type") == "alignment_done":
                fragment.native_done = True
                fragment.native_unverified |= event.get("completion_verified") is False
                self._condition.notify_all()

    def _spans(self, fragment):
        normalized, mapping = _normalized_map(fragment.text)
        spans = []
        for key, field, level in (("words", "word", "word"), ("characters", "character", "character")):
            cursor = 0
            for item in sorted(fragment.items[key], key=lambda item: item["start_time"]):
                text = str(item.get(field, item.get("text", "")))
                needle, _ = _normalized_map(text)
                needle = needle.strip()
                position = normalized.find(needle, cursor) if needle else -1
                start = end = None
                if position >= 0:
                    first, last = mapping[position][0], mapping[position + len(needle) - 1][1]
                    cursor = position + len(needle)
                    if fragment.ranges:
                        mapped = fragment.ranges[first:last]
                        if mapped and all(a is not None and b is not None for a, b in mapped):
                            first, last = min(a for a, _ in mapped), max(b for _, b in mapped)
                        else:
                            first = last = None
                    if first is not None and fragment.text_offset is not None:
                        start, end = fragment.text_offset + first, fragment.text_offset + last
                try:
                    spans.append(AlignedSpan(
                        text=text, start_time=max(0.0, fragment.start + float(item["start_time"])),
                        end_time=max(0.0, fragment.start + float(item["end_time"])),
                        text_start=start, text_end=end,
                        source="native" if fragment.native and not getattr(fragment.engine, "streaming_aligner", None) else "model",
                        state="final" if fragment.finished or fragment.native else "provisional",
                        level=level, score=item.get("score"), fragment_id=fragment.id,
                    ))
                except (KeyError, TypeError, ValueError):
                    LOGGER.warning("Ignoring invalid alignment span")
        return spans

    def _coverage(self):
        expected = sum(f.expected for f in self._fragments.values())
        return sum(f.covered for f in self._fragments.values()) / expected if expected else 1.0

    def _publish(self, fragment):
        if fragment.generation != self._generation:
            return
        spans = self._spans(fragment)
        fragment.covered, fragment.expected = coverage_counts(
            fragment.text, (span.text for span in spans if span.level == self.config.level)
        )
        self.timeline.publish(
            spans, fragment_id=fragment.id, status="aligning", coverage=self._coverage()
        )

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or self._pending or self._finishing, timeout=0.1
                )
                if self._closed:
                    return
                if not self._pending:
                    if self._finishing:
                        unfinished = [f for f in self._fragments.values() if not f.done.is_set()]
                        for fragment in unfinished:
                            if not fragment.native:
                                continue
                            session_done = getattr(fragment.native_session, "done", None)
                            if session_done is not None and session_done.is_set():
                                fragment.done.set()
                                self._check_native_result(fragment)
                            elif time.monotonic() - (fragment.finished_at or time.monotonic()) > 30:
                                self._fail(fragment, "Native alignment did not finish within 30 seconds")
                        if all(f.done.is_set() for f in self._fragments.values()):
                            coverage = self._coverage()
                            status = ("unavailable" if self.timeline.snapshot().errors else
                                      "partial" if coverage < 1.0 or any(
                                          f.native_unverified for f in self._fragments.values()) else "ready")
                            self.timeline.finish(status=status, coverage=coverage)
                            self._finishing = False
                        else:
                            self._condition.wait(timeout=0.05)
                    continue
                _, fragment = self._pending.popitem(last=False)
                if fragment.unavailable or fragment.generation != self._generation:
                    continue
                text, version = fragment.text, fragment.version
                final = fragment.finished
                audio = np.concatenate(fragment.chunks) if fragment.chunks else np.empty(0, np.float32)
                audio_start = fragment.audio_start
            try:
                if self._model_error is not None:
                    raise RuntimeError(self._model_error)
                if self._model is None:
                    self._model = make_alignment_model(self.config, fragment.engine)
                if not text.strip() or not len(audio):
                    result = {"words": [], "characters": []}
                elif isinstance(self._model, CPUAlignmentModel):
                    if fragment.cursor is None:
                        fragment.cursor = self._model.create_cursor(level=self.config.level)
                    result = fragment.cursor.align(text, audio, fragment.rate, audio_start=audio_start)
                else:
                    result = align_snapshot(self._model, text, audio, fragment.rate, self.config.level)
                with self._condition:
                    if fragment.generation != self._generation or fragment.unavailable:
                        continue
                    key = "characters" if self.config.level == "character" else "words"
                    if final and text.strip() and not result.get(key):
                        self._fail(fragment, "Alignment model returned no timestamps for the requested level")
                        continue
                    fragment.items = result
                    self._publish(fragment)
                    if fragment.cursor is not None:
                        discard = fragment.cursor.audio_start - fragment.audio_start
                        while discard > 0 and fragment.chunks:
                            count = min(discard, len(fragment.chunks[0]))
                            fragment.chunks[0] = fragment.chunks[0][count:]
                            if not len(fragment.chunks[0]):
                                fragment.chunks.pop(0)
                            fragment.audio_start += count
                            discard -= count
                    if final and version == fragment.version:
                        fragment.chunks.clear()
                        fragment.done.set()
            except Exception as exc:
                LOGGER.warning("Shared alignment unavailable: %s", exc)
                with self._condition:
                    if fragment.generation == self._generation:
                        self._fail(fragment, str(exc))
            finally:
                with self._condition:
                    self._condition.notify_all()

    def finish(self, *, cancelled=False):
        if cancelled:
            self.cancel()
            return
        with self._condition:
            self._finishing = True
            self._condition.notify_all()

    def cancel(self):
        with self._condition:
            self._generation += 1
            self._pending.clear()
            self._finishing = False
            self._playback_active = False
            for fragment in self._fragments.values():
                fragment.chunks.clear()
                fragment.done.set()
            self.timeline.cancel()
            self._condition.notify_all()

    def before_playback(self, position):
        if self.config.mode != "synced":
            return
        deadline = time.monotonic() + self.config.max_buffer_seconds
        with self._condition:
            candidates = [f for f in self._fragments.values()
                          if f.start <= position + 1e-6
                          and position < f.start + f.samples / f.rate + 1e-6]
            if not candidates:
                return
            fragment = candidates[-1]
            if fragment.id in self._sync_waited:
                return
            self._sync_waited.add(fragment.id)
            while not any(w.fragment_id == fragment.id for w in self.timeline.words):
                if fragment.done.is_set() or self.timeline.snapshot().final:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.timeline.set_status("behind")
                    return
                self._condition.wait(min(remaining, 0.02))

    def advance(self, position):
        with self._condition:
            self._position = float(position)
            self._playback_active = True
            self.timeline.seek(self._position)
        self._deliver()

    def playback_finished(self):
        with self._condition:
            self._playback_active = False

    def _deliver(self):
        callbacks = []
        with self._condition:
            if not self._playback_active:
                return
            snapshot = self.timeline.snapshot()
            for spans, callback in ((snapshot.words, self._on_word), (snapshot.characters, self._on_character)):
                if callback is None:
                    continue
                for span in spans:
                    key = (span.fragment_id, span.level, span.text_start, span.text_end, span.text)
                    if span.text_start is None:
                        key += (span.start_time,)
                    if span.start_time <= self._position + 1e-8 and key not in self._fired:
                        self._fired.add(key)
                        callbacks.append((callback, span))
        for callback, span in callbacks:
            try:
                callback(span)
            except Exception:
                LOGGER.exception("Alignment playback callback failed")

    def _updated(self, snapshot):
        # Timeline updates can originate on the audio producer. Coalesce UI
        # snapshots and invoke user code outside producer/controller locks.
        if hasattr(self, "_notification"):
            self._latest_update = (self._generation, snapshot)
            self._notification.set()

    def _notify(self):
        while not self._closed:
            self._notification.wait(0.1)
            self._notification.clear()
            update, self._latest_update = self._latest_update, None
            if update is None or update[0] != self._generation:
                continue
            if self._on_update is not None:
                try:
                    self._on_update(update[1])
                except Exception:
                    LOGGER.exception("Alignment update callback failed")
            self._deliver()

    def close(self):
        self.cancel()
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._binding:
            self._binding.detach()
        if threading.current_thread() is not self._worker:
            self._worker.join(timeout=1.0)
