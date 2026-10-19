"""Shared alignment wiring for the existing PCM HTTP/WebSocket services."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from .controller import AlignmentController
from .timeline import AlignmentConfig


_ALIGNMENT_TIME_KEYS = frozenset({
    "start_time", "end_time", "time_start", "time_end", "duration",
    "global_start_time", "global_end_time", "engine_audio_offset",
    "trim_end_time", "audio_duration", "audio_duration_s",
    "pcm_start_sample", "audio_start_sample", "start_sample", "end_sample",
})


def scale_alignment_event(event, speed):
    """Map raw engine timestamps onto the tempo-processed PCM timeline."""
    try:
        factor = 1.0 / float(speed)
    except (TypeError, ValueError, ZeroDivisionError):
        return event
    if factor == 1.0:
        return event

    def convert(value, key=None):
        if isinstance(value, Mapping):
            result = {}
            for name, item in value.items():
                converted = convert(item)
                if (
                    name in _ALIGNMENT_TIME_KEYS
                    and isinstance(converted, (int, float))
                    and not isinstance(converted, bool)
                ):
                    converted *= factor
                result[name] = converted
            return result
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        return value

    return convert(event)


def parse_alignment_request(value, engine, language):
    if isinstance(value, dict):
        allowed = {"source", "mode", "level", "max_buffer_seconds"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError("Unsupported alignment fields: " + ", ".join(sorted(unknown)))
    config = AlignmentConfig.coerce(value, language=language)
    if config is None:
        return None
    if config.max_buffer_seconds > 1.0:
        raise ValueError("alignment.max_buffer_seconds must not exceed 1 second")
    native = bool(getattr(engine, "provides_word_timings", False))
    native_backend = getattr(engine, "streaming_aligner", None)
    studio_backend = getattr(engine, "studio_aligner", None)
    native_level = config.level == "word" or config.level in getattr(
        native_backend, "timestamp_levels", ()
    )
    studio_levels = getattr(studio_backend, "timestamp_levels", None) or (
        "word", "character"
    )
    if config.source == "native" and not (native and native_level):
        raise ValueError("Engine does not provide the requested native timestamp level")
    if config.source == "native":
        return replace(config, aligner=native_backend)
    if studio_backend is not None and config.level in studio_levels:
        return replace(config, source="model", aligner=studio_backend)
    if config.source == "model":
        if native_backend is None:
            raise ValueError("Configure an alignment backend on the server before requesting alignment")
        return replace(config, aligner=native_backend)
    if not (native and native_level) and native_backend is None:
        raise ValueError("Configure an alignment backend on the server before requesting alignment")
    return replace(config, aligner=native_backend)


class ServerAlignment:
    def __init__(self, engine, request, state):
        self.engine, self.state = engine, state
        self.original_callback = getattr(engine, "on_alignment", None)
        self.original_gate = getattr(engine, "alignment_start_event", None)
        self.original_cancel = getattr(engine, "alignment_cancel_event", None)
        self.original_streaming_aligner = getattr(engine, "streaming_aligner", None)
        self._native_alignment_suppressed = False
        state.alignment_done.clear()
        config = replace(request.alignment, language=request.language)

        def updated(snapshot):
            if state.alignment_closed.is_set() or state.cancelled.is_set():
                return
            if snapshot.final or state.alignment_events.qsize() < 8:
                state.alignment_events.put(scale_alignment_event({
                    **snapshot.to_dict(), "type": "alignment",
                    "time_base": "fragment", "text_offset_unit": "unicode_codepoint",
                }, request.speed))
            if snapshot.final:
                state.alignment_events.put({
                    "type": "alignment_done", "complete": snapshot.status == "ready",
                    "finish_reason": snapshot.status, "coverage": snapshot.coverage,
                })
                state.alignment_done.set()

        self.controller = AlignmentController(config, on_update=updated)
        state.alignment_session = self.controller
        source = request.source_text if request.source_text is not None else request.text
        self.controller.append_source(source)
        if hasattr(engine, "on_alignment"):
            original = self.original_callback

            def controls(event):
                kind = event.get("type")
                verified_stop = (kind == "alignment_complete"
                                 and event.get("completion_verified") is True
                                 and event.get("stop_requested") is True)
                if (not state.alignment_closed.is_set() and not state.cancelled.is_set()
                        and (verified_stop or kind not in {"alignment", "alignment_done", "alignment_complete"})):
                    # Display snapshots replace timing updates, but must not swallow
                    # the engine-authorized cutoff used by queued playback.
                    state.alignment_events.put(scale_alignment_event(event, request.speed))
                if original is not None:
                    original(event)

            engine.on_alignment = controls
        if hasattr(engine, "alignment_start_event"):
            engine.alignment_start_event = state.first_pcm_published if state.response_format == "pcm" else None
        if hasattr(engine, "alignment_cancel_event"):
            engine.alignment_cancel_event = state.cancelled
        self.binding = self.controller.bind(engine)
        self.binding.begin(request.text, original_text=source, source_ranges=request.source_ranges)
        active_guard = bool(
            getattr(engine, "hallucination_guard", False)
            or getattr(engine, "acoustic_guard", False)
        )
        if (config.source == "model" and self.original_streaming_aligner is not None
                and not active_guard):
            # The display model owns this request when no engine guard needs the
            # native alignment stream for an authorized completion cutoff.
            engine.streaming_aligner = None
            self._native_alignment_suppressed = True

    def finish(self, *, success):
        try:
            self.binding.finish(success=success)
            if self.state.cancelled.is_set():
                self.controller.cancel()
                self.state.alignment_done.set()
            else:
                self.controller.finish()
        finally:
            self.binding.detach()
            if hasattr(self.engine, "on_alignment"):
                self.engine.on_alignment = self.original_callback
            if hasattr(self.engine, "alignment_start_event"):
                self.engine.alignment_start_event = self.original_gate
            if hasattr(self.engine, "alignment_cancel_event"):
                self.engine.alignment_cancel_event = self.original_cancel
            if self._native_alignment_suppressed:
                self.engine.streaming_aligner = self.original_streaming_aligner
