from __future__ import annotations

import queue
import threading
import time

import pytest

from RealtimeTTS.engines.language_routing_engine import (
    LanguageRoute,
    LanguageRoutingEngine,
    SUPPORTED_STREAM_INFO,
    normalize_language,
)


class FakeEngine:
    def __init__(
        self,
        name,
        *,
        chunks=(),
        result=True,
        direct_voice_control=False,
        language_aliases=False,
        block_first=False,
        emit_after_stop=False,
        provides_word_timings=False,
        defer_alignment=False,
        alignment_ranges=None,
        supports_alignment_ranges=False,
    ):
        self.engine_name = name
        self.queue = queue.Queue()
        self.timings = queue.Queue()
        self.chunks = list(chunks)
        self.result = result
        self.direct_voice_control = direct_voice_control
        self.instruction_control = direct_voice_control
        self.provides_word_timings = provides_word_timings
        self.defer_alignment = defer_alignment
        self._alignment_ranges = tuple(
            {"start": span["start"], "end": span["end"]}
            for span in (alignment_ranges or ())
        )
        if not supports_alignment_ranges:
            self.alignment_non_speech_ranges = None
        self.captured_alignments = []
        self.on_alignment = None
        self.last_alignment = None
        self.language_aliases = language_aliases
        self.block_first = block_first
        self.emit_after_stop = emit_after_stop
        self.started = threading.Event()
        self.release = threading.Event()
        self.stop_called = threading.Event()
        self.calls = []
        self.synthesis_count = 0
        self._context_serial = 0
        self._stop = threading.Event()
        self.audio_duration = 0
        self.can_consume_generators = False

    def get_stream_info(self):
        return SUPPORTED_STREAM_INFO

    def alignment_non_speech_ranges(self, text):
        del text
        return [dict(span) for span in self._alignment_ranges]

    def set_inline_voice(self, voice):
        self.calls.append(("voice", voice))

    def set_voice(self, voice):
        self.calls.append(("voice", voice))

    def set_language(self, language):
        self.calls.append(("language", language))
        if self.language_aliases and language == "en":
            raise ValueError("short key not accepted")

    def set_instructions(self, instructions):
        self.calls.append(("instructions", instructions))

    def create_stream_context(self):
        self._context_serial += 1
        context = f"context-{self.engine_name}-{self._context_serial}"
        self.calls.append(("create_context", context))
        return context

    def set_stream_context(self, context):
        self.calls.append(("set_context", context))

    def close_stream_context(self, context):
        self.calls.append(("close_context", context))

    def synthesize(self, text, sentence_count=0):
        self.synthesis_count += 1
        self.started.set()
        if self.block_first and self.synthesis_count == 1:
            self.release.wait(2)
        if self._stop.is_set() and not self.emit_after_stop:
            return False
        for chunk in self.chunks:
            self.queue.put(chunk)
        self.audio_duration = 0.25
        if self.provides_word_timings and self.on_alignment is not None:
            self.last_alignment = object()
            callback = self.on_alignment
            if self.defer_alignment:
                self.captured_alignments.append(callback)
            else:
                callback(
                    {"type": "alignment_done", "engine_audio_offset": 0.0}
                )
        return self.result

    def stop(self):
        self.stop_called.set()
        self._stop.set()
        self.release.set()

    def pause(self, timeout=2.0):
        self.calls.append(("pause", timeout))
        return True

    def resume(self):
        self.calls.append(("resume",))
        return True


def make_router(*entries, **kwargs):
    return LanguageRoutingEngine({"en": list(entries)}, **kwargs)


def drain(output):
    result = []
    while True:
        try:
            result.append(output.get_nowait())
        except queue.Empty:
            return result


def test_language_normalization_and_wildcard_selection():
    english = FakeEngine("english")
    explicit_fallback = FakeEngine("fallback", chunks=[b"fallback"])
    wildcard = FakeEngine("wildcard", chunks=[b"wildcard"])
    router = LanguageRoutingEngine(
        {"en": [english, explicit_fallback], "*": [wildcard]},
        pre_first_pcm_timeout=0.05,
    )
    router.set_language("EN_us")
    assert router.get_language() == "en"
    assert normalize_language("Chinese") == "zh"
    assert router.synthesize("hello") is True
    assert drain(router.queue) == [b"fallback"]
    assert wildcard.synthesis_count == 0

    wildcard_only = LanguageRoutingEngine(
        {"en": [FakeEngine("english")], "*": [wildcard]},
        default_language="de",
        pre_first_pcm_timeout=0.05,
    )
    assert wildcard_only.synthesize("hola") is True
    assert drain(wildcard_only.queue) == [b"wildcard"]


def test_alignment_non_speech_ranges_follow_current_and_default_language():
    english = FakeEngine(
        "english",
        alignment_ranges=[{"start": 3, "end": 10}],
        supports_alignment_ranges=True,
    )
    spanish = FakeEngine(
        "spanish",
        alignment_ranges=[{"start": 4, "end": 11}],
        supports_alignment_ranges=True,
    )
    router = LanguageRoutingEngine({"en": [english], "es": [spanish]})
    assert router.alignment_non_speech_ranges("😀 (laugh)") == [
        {"start": 3, "end": 10}
    ]
    router.set_language("es")
    assert router.alignment_non_speech_ranges("😀 (laugh)") == [
        {"start": 4, "end": 11}
    ]
    router.set_language("auto")
    assert router.alignment_non_speech_ranges("😀 (laugh)") == [
        {"start": 3, "end": 10}
    ]
    assert english.synthesis_count == spanish.synthesis_count == 0


def test_alignment_non_speech_ranges_require_identical_fallback_spans():
    ranges = [{"start": 3, "end": 10}]
    first = FakeEngine(
        "first", alignment_ranges=ranges, supports_alignment_ranges=True
    )
    same = FakeEngine(
        "same", alignment_ranges=ranges, supports_alignment_ranges=True
    )
    router = LanguageRoutingEngine({"en": [first, same]})
    assert router.alignment_non_speech_ranges("text") == ranges

    mismatch = FakeEngine(
        "mismatch",
        alignment_ranges=[{"start": 4, "end": 11}],
        supports_alignment_ranges=True,
    )
    router = LanguageRoutingEngine({"en": [first, mismatch]})
    assert router.alignment_non_speech_ranges("text") == []

    qwen = FakeEngine("qwen")
    router = LanguageRoutingEngine({"en": [first, qwen]})
    assert router.alignment_non_speech_ranges("text") == []
    assert first.synthesis_count == mismatch.synthesis_count == qwen.synthesis_count == 0


def test_route_voice_and_sticky_instruction_are_provider_local():
    english = FakeEngine("english", direct_voice_control=True, chunks=[b"en"])
    chinese = FakeEngine("chinese", direct_voice_control=True, chunks=[b"zh"])
    router = LanguageRoutingEngine(
        {
            "en": [LanguageRoute(english, "en-default", {"logical": "en-logical"})],
            "zh": [LanguageRoute(chinese, "zh-default", {"logical": "zh-logical"})],
        }
    )
    router.set_voice("logical")
    router.set_instructions("warm")
    assert router.synthesize("one")
    router.set_language("Chinese")
    assert router.synthesize("two")
    assert ("voice", "en-logical") in english.calls
    assert ("voice", "zh-logical") in chinese.calls
    assert ("instructions", "warm") in english.calls
    assert ("instructions", "warm") in chinese.calls
    assert ("voice", "en-logical") not in chinese.calls
    assert ("voice", "zh-logical") not in english.calls


def test_stable_output_order_and_provider_queue_restore():
    provider = FakeEngine("ordered", chunks=[b"a", b"b"])
    original_queue, original_timings = provider.queue, provider.timings
    router = make_router(provider)
    assert router.synthesize("text")
    assert drain(router.queue) == [b"a", b"b"]
    assert provider.queue is original_queue
    assert provider.timings is original_timings


def test_pre_pcm_timeout_uses_fallback_provider():
    first = FakeEngine("slow", block_first=True)
    second = FakeEngine("ready", chunks=[b"ready"])
    router = make_router(
        first, second, pre_first_pcm_timeout=0.05, stop_timeout=0.5
    )
    assert router.synthesize("text")
    assert drain(router.queue) == [b"ready"]
    assert first.stop_called.is_set()
    assert second.synthesis_count == 1


def test_finite_fallback_before_pcm_and_no_fallback_after_pcm():
    first = FakeEngine("first", result=False)
    second = FakeEngine("second", chunks=[b"second"])
    router = make_router(first, second, pre_first_pcm_timeout=0.05)
    assert router.synthesize("text")
    assert drain(router.queue) == [b"second"]
    assert first.synthesis_count == second.synthesis_count == 1

    committed = FakeEngine("committed", chunks=[b"first"], result=False)
    never = FakeEngine("never", chunks=[b"second"])
    router = make_router(committed, never, pre_first_pcm_timeout=0.05)
    with pytest.raises(RuntimeError, match="after producing PCM"):
        router.synthesize("text")
    assert never.synthesis_count == 0
    assert drain(router.queue) == [b"first"]


def test_cancel_fences_late_output_and_router_reuses_provider():
    provider = FakeEngine(
        "cancel", chunks=[b"late"], block_first=True, emit_after_stop=True
    )
    router = make_router(provider, pre_first_pcm_timeout=1.0, stop_timeout=0.5)
    result = {}
    worker = threading.Thread(
        target=lambda: result.setdefault("value", router.synthesize("first"))
    )
    worker.start()
    assert provider.started.wait(1)
    router.stop()
    worker.join(1)
    assert not worker.is_alive()
    assert result["value"] is False
    assert provider.stop_called.is_set()
    assert drain(router.queue) == []

    provider.block_first = False
    provider._stop.clear()
    provider.chunks = [b"new"]
    provider.started.clear()
    assert router.synthesize("second") is True
    assert drain(router.queue) == [b"new"]


def test_pause_and_resume_forward_to_active_provider():
    provider = FakeEngine("pausable", chunks=[b"ok"], block_first=True)
    router = make_router(provider, pre_first_pcm_timeout=1.0)
    worker = threading.Thread(target=lambda: router.synthesize("text"))
    worker.start()
    assert provider.started.wait(1)
    assert router.pause() is True
    assert router.resume() is True
    router.stop()
    worker.join(1)
    assert not worker.is_alive()
    assert ("pause", 2.0) in provider.calls
    assert ("resume",) in provider.calls


def test_stream_contexts_forward_and_close():
    provider = FakeEngine("context", chunks=[b"x"])
    router = make_router(provider)
    context = router.create_stream_context()
    router.set_stream_context(context)
    assert router.synthesize("text")
    router.close_stream_context(context)
    assert ("create_context", "context-context-1") in provider.calls
    assert ("set_context", "context-context-1") in provider.calls
    assert ("close_context", "context-context-1") in provider.calls


def test_voice_parameters_allow_controls_and_reject_sampling():
    router = make_router(FakeEngine("controls"))
    router.set_voice_parameters(language="English", instructions="")
    assert router.get_language() == "en"
    with pytest.raises(ValueError, match="provider parameters"):
        router.set_voice_parameters(temperature=0.5)


def test_routing_rejects_wildcard_default_and_duplicate_canonical_routes():
    provider = FakeEngine("provider")
    with pytest.raises(ValueError, match="default_language"):
        LanguageRoutingEngine({"en": [provider]}, default_language="*")
    with pytest.raises(ValueError, match="duplicate route"):
        LanguageRoutingEngine({"en": [provider], "English": [provider]})


def test_routing_rejects_nonfinite_timeouts_and_generators():
    with pytest.raises(ValueError, match="finite"):
        LanguageRoutingEngine({"en": [FakeEngine("x")]}, pre_first_pcm_timeout=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        LanguageRoutingEngine({"en": [FakeEngine("x")]}, stop_timeout=float("inf"))
    provider = FakeEngine("generator")
    provider.can_consume_generators = True
    router = make_router(provider)
    assert router.can_consume_generators is False


def test_format_mismatch_is_rejected():
    class OtherFormat(FakeEngine):
        def get_stream_info(self):
            return (8, 2, 24000)

    with pytest.raises(ValueError, match="requires"):
        make_router(FakeEngine("ok"), OtherFormat("bad"))


def test_instructions_are_not_sent_to_non_direct_provider():
    provider = FakeEngine("no-instructions", chunks=[b"x"])
    router = make_router(provider)
    router.set_instructions("ignored")
    assert router.synthesize("text")
    assert not any(kind == "instructions" for kind, _ in provider.calls)


def test_native_alignment_bridge_is_strict_and_fenced():
    provider = FakeEngine(
        "native", chunks=[b"x"], provides_word_timings=True,
        defer_alignment=True,
    )
    router = make_router(provider)
    router.audio_duration = 1.0
    events = []
    router.on_alignment = events.append
    assert router.provides_word_timings is True
    assert router.synthesize("text")
    assert events == []
    provider.captured_alignments[0](
        {"type": "alignment_done", "engine_audio_offset": 0.0}
    )
    assert events == [
        {"type": "alignment_done", "engine_audio_offset": 1.0}
    ]
    assert router.last_alignment is provider.last_alignment
    router.stop()
    provider.captured_alignments[0](
        {"type": "alignment_done", "engine_audio_offset": 0.0}
    )
    assert len(events) == 1


def test_voice_mapping_rejects_unknown_logical_voice_and_falls_back():
    first = FakeEngine("mapped", chunks=[b"first"])
    second = FakeEngine("fallback", chunks=[b"second"])
    router = LanguageRoutingEngine({
        "en": [
            LanguageRoute(first, voices={"warm": "provider-warm"}),
            second,
        ]
    })
    router.set_voice("missing")
    assert router.synthesize("text") is True
    assert drain(router.queue) == [b"second"]
    assert first.synthesis_count == 0
    assert second.synthesis_count == 1


def test_native_alignment_callbacks_are_per_attempt_and_not_duplicated():
    provider = FakeEngine(
        "native", chunks=[b"x"], provides_word_timings=True, defer_alignment=True
    )
    router = make_router(provider)
    first_events, second_events = [], []
    router.on_alignment = first_events.append
    assert router.synthesize("first") is True
    router.on_alignment = second_events.append
    assert router.synthesize("second") is True
    assert len(provider.captured_alignments) == 2

    provider.captured_alignments[0](
        {"type": "alignment_done", "engine_audio_offset": 1.0}
    )
    provider.captured_alignments[1](
        {"type": "alignment_done", "engine_audio_offset": 2.0}
    )
    assert first_events == [
        {"type": "alignment_done", "engine_audio_offset": 1.0}
    ]
    assert len(second_events) == 1
    assert second_events[0]["type"] == "alignment_done"
    assert second_events[0]["engine_audio_offset"] != 2.0
    provider.captured_alignments[0](
        {"type": "alignment_done", "engine_audio_offset": 9.0}
    )
    assert len(first_events) == 1


def test_provider_language_fallback_uses_one_canonical_alias():
    provider = FakeEngine("alias", chunks=[b"x"], language_aliases=True)
    router = make_router(provider)
    assert router.synthesize("text")
    assert ("language", "en") in provider.calls
    assert ("language", "english") in provider.calls


def test_context_creation_and_close_are_scoped_to_each_stream():
    provider = FakeEngine(
        "contexts", chunks=[b"x"], provides_word_timings=True, defer_alignment=True
    )
    router = make_router(provider)
    first_events, second_events = [], []
    first_context = router.create_stream_context()
    assert router._stream_context is None
    first_child = first_context.contexts[id(provider)]
    router.set_stream_context(first_context)
    router.on_alignment = first_events.append
    assert router.synthesize("first")
    second_context = router.create_stream_context()
    assert router._stream_context is first_context
    second_child = second_context.contexts[id(provider)]
    assert first_child != second_child
    router.set_stream_context(second_context)
    router.on_alignment = second_events.append
    assert router.synthesize("second")

    router.close_stream_context(first_context)
    assert ("close_context", first_child) in provider.calls
    assert router._stream_context is second_context
    provider.captured_alignments[0](
        {"type": "alignment_done", "engine_audio_offset": 0.0}
    )
    assert first_events == []
    provider.captured_alignments[1](
        {"type": "alignment_done", "engine_audio_offset": 0.0}
    )
    assert len(second_events) == 1

    router.close_stream_context(second_context)
    assert ("close_context", second_child) in provider.calls
    provider.captured_alignments[1](
        {"type": "alignment_done", "engine_audio_offset": 1.0}
    )
    assert len(second_events) == 1
    assert router._stream_context is None


def test_cancel_after_pcm_does_not_wait_for_uncooperative_provider():
    class UncooperativeAfterPcm(FakeEngine):
        def __init__(self):
            super().__init__("after-pcm")
            self.exited = threading.Event()

        def synthesize(self, text, sentence_count=0):
            try:
                self.synthesis_count += 1
                self.started.set()
                self.queue.put(b"first")
                self.release.wait(2.0)
                return True
            finally:
                self.exited.set()

        def stop(self):
            self.stop_called.set()
            self._stop.set()

    provider = UncooperativeAfterPcm()
    router = make_router(
        provider, pre_first_pcm_timeout=1.0, stop_timeout=0.02
    )
    result = {}
    worker = threading.Thread(
        target=lambda: result.setdefault("value", router.synthesize("text"))
    )
    worker.start()
    try:
        assert provider.started.wait(1.0)
        assert router._active_attempt is not None
        assert router._active_attempt.first_pcm.wait(1.0)
        router.stop()
        worker.join(0.5)
        assert not worker.is_alive()
        assert result["value"] is False
        assert provider.stop_called.is_set()
    finally:
        provider.release.set()
        worker.join(1.0)
    assert not worker.is_alive()
    assert provider.exited.wait(1.0)


def test_timeout_after_pcm_commit_is_terminal_without_fallback():
    class GateQueue(queue.Queue):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def put(self, item, block=True, timeout=None):
            super().put(item, block=False)
            self.entered.set()
            self.release.wait(1.0)

    gate = GateQueue()

    class GateStoppingEngine(FakeEngine):
        def stop(self):
            self.stop_called.set()
            self._stop.set()
            gate.release.set()

    first = GateStoppingEngine("first", chunks=[b"first"])
    fallback = FakeEngine("fallback", chunks=[b"fallback"])
    router = make_router(
        first,
        fallback,
        pre_first_pcm_timeout=0.01,
        stop_timeout=0.2,
    )
    router.queue = gate
    try:
        with pytest.raises(RuntimeError, match="timed out while delivering PCM"):
            router.synthesize("text")
    finally:
        gate.release.set()
    assert gate.entered.is_set()
    assert first.synthesis_count == 1
    assert fallback.synthesis_count == 0
    assert drain(router.queue) == [b"first"]


def test_uncooperative_timeout_quarantines_provider_until_worker_exit():
    class Uncooperative(FakeEngine):
        def __init__(self):
            super().__init__("stalled", block_first=True)
            self.exited = threading.Event()

        def synthesize(self, text, sentence_count=0):
            try:
                return super().synthesize(text, sentence_count)
            finally:
                self.exited.set()

        def stop(self):
            self.stop_called.set()
            self._stop.set()

    stalled = Uncooperative()
    fallback = FakeEngine("fallback", chunks=[b"fallback"])
    original_queue, original_timings = stalled.queue, stalled.timings
    router = make_router(
        stalled, fallback, pre_first_pcm_timeout=0.02, stop_timeout=0.01
    )
    with pytest.raises(TimeoutError, match="did not stop"):
        router.synthesize("first")
    assert stalled.stop_called.is_set()
    assert stalled.queue is not original_queue
    assert stalled.timings is not original_timings

    assert router.synthesize("second")
    assert drain(router.queue) == [b"fallback"]
    assert stalled.synthesis_count == 1

    stalled.release.set()
    assert stalled.exited.wait(1.0)
    deadline = time.monotonic() + 1.0
    while stalled.queue is not original_queue and time.monotonic() < deadline:
        time.sleep(0.01)
    assert stalled.queue is original_queue
    assert stalled.timings is original_timings
