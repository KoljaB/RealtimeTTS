from __future__ import annotations

import threading

import pytest

from RealtimeTTS.inline_markup import InlineMarkupParser
from tests.test_inline_language import ObservingEngine, _coalesce, stream_factory
from tests.test_text_to_stream_alignment import _PCMEngine


@pytest.mark.parametrize(("method", "registry"), [
    ("add_instructions", "instruction_tags"),
    ("add_vocalizations", "vocalization_tags"),
])
def test_bulk_registration_merges_copies_and_chains(stream_factory, method, registry):
    from types import MappingProxyType
    with stream_factory() as stream:
        add = getattr(stream, method)
        assert add({"first": "Old"}) is stream
        values = {"first": "Updated", "second": "Second"}
        assert add(MappingProxyType(values)) is stream
        values["second"] = "Changed outside the stream"
        assert getattr(stream, registry) == {"first": "Updated", "second": "Second"}
        assert add({}) is stream


@pytest.mark.parametrize(("method", "registry"), [
    ("add_instructions", "instruction_tags"),
    ("add_vocalizations", "vocalization_tags"),
])
def test_bulk_registration_rejects_batch_without_partial_changes(stream_factory, method, registry):
    with stream_factory() as stream:
        add = getattr(stream, method)
        add({"existing": "Original"})
        for invalid, error in [
            ({"valid": "New", "pause": "Conflicting"}, ValueError),
            ({"valid": "New", "invalid": 123}, TypeError),
            ({"valid": "New", "VALID": "Ambiguous"}, ValueError),
            ([("valid", "New")], TypeError),
        ]:
            with pytest.raises(error):
                add(invalid)
            assert getattr(stream, registry) == {"existing": "Original"}


def test_bulk_vocalizations_accept_optional_directions(stream_factory):
    with stream_factory() as stream:
        assert stream.add_vocalizations({"sigh": None, "laugh": "Brief laughter"}) is stream
        assert stream.vocalization_tags == {"sigh": None, "laugh": "Brief laughter"}
        with pytest.raises(TypeError, match="string"):
            stream.add_instructions({"normal": None})
        assert stream.instruction_tags == {}


class InstructionEngine(ObservingEngine):
    def post_init(self):
        super().post_init()
        self.instructions = "baseline"
        self.deliveries = []

    def set_instructions(self, value):
        self.instructions = value

    def synthesize(self, text, sentence_count=0):
        if not isinstance(text, str):
            text = "".join(text)
        self.deliveries.append((text.strip(), self.instructions, self.language))
        return super().synthesize(text, sentence_count)


def test_instruction_parser_survives_every_chunk_boundary_and_preserves_unknown():
    source = "First.[surprised]Wow!<whispered>Secret.[unknown]Literal."
    expected = [
        ("text", "First."), ("instruction", "Excited"),
        ("text", "Wow!"), ("instruction", "Whisper"),
        ("text", "Secret.[unknown]Literal."),
    ]
    for split in range(len(source) + 1):
        parser = InlineMarkupParser(
            1.0, instruction_tags={"surprised": "Excited", "whispered": "Whisper"})
        assert _coalesce(parser.iter_actions((source[:split], source[split:]))) == expected


@pytest.mark.parametrize("tags", [
    {"en": "x"}, {"speed": "x"}, {"pause": "x"}, {"with space": "x"},
    {"normal": "x", "NORMAL": "y"}, {"": "x"},
])
def test_ambiguous_instruction_registration_is_rejected(tags):
    with pytest.raises(ValueError):
        InlineMarkupParser(1.0, instruction_tags=tags)


def test_instruction_registration_rejects_voice_collision_and_non_string(stream_factory):
    stream = stream_factory()
    stream.add_voice("whispered", "voice")
    with pytest.raises(ValueError, match="conflicting"):
        stream.add_instruction("WHISPERED", "Whisper")
    with pytest.raises(TypeError, match="string"):
        stream.add_instruction("normal", None)


@pytest.mark.parametrize("generator", [False, True])
def test_instruction_boundaries_language_persistence_and_next_play_reset(stream_factory, generator):
    engine = InstructionEngine()
    engine.can_consume_generators = generator
    stream = stream_factory(engine, language_tags=True)
    stream.add_instruction("surprised", "Excited").add_instruction("whispered", "Whisper")
    stream.add_instruction("normal", "")
    stream.feed("First.[surprised]Wow!<es>Hola.</es>[whispered]Secret.[normal]Done.").play()
    assert engine.deliveries == [
        ("First.", "baseline", "en"), ("Wow!", "Excited", "en"),
        ("Hola.", "Excited", "es"), ("Secret.", "Whisper", "en"),
        ("Done.", "", "en"),
    ]
    assert engine.instructions == "baseline"
    stream.feed("Again.").play()
    assert engine.deliveries[-1] == ("Again.", "baseline", "en")


@pytest.mark.parametrize("generator", [False, True])
def test_instruction_only_stream_and_restoration_without_language_controls(stream_factory, generator):
    engine = InstructionEngine()
    engine.can_consume_generators = generator
    stream = stream_factory(engine).add_instruction("whispered", "Whisper")
    stream.feed("[whispered]Secret.").play()
    assert engine.deliveries == [("Secret.", "Whisper", "en")]
    assert engine.instructions == "baseline"
    assert not stream.play_lock.locked()


@pytest.mark.parametrize("generator", [False, True])
def test_late_feed_keeps_delivery_until_external_play_finishes(stream_factory, generator):
    class Blocking(InstructionEngine):
        def post_init(self):
            super().post_init()
            self.started, self.release = threading.Event(), threading.Event()
        def synthesize(self, text, sentence_count=0):
            if not isinstance(text, str):
                text = "".join(text)
            if text.strip() == "Inside.":
                self.started.set()
                assert self.release.wait(2)
            return super().synthesize(text, sentence_count)
    engine = Blocking()
    engine.can_consume_generators = generator
    stream = stream_factory(engine).add_instruction("whispered", "Whisper")
    try:
        stream.feed("[whispered]Inside.").play_async()
        assert engine.started.wait(2)
        stream.feed("Tail.")
        engine.release.set()
        stream.play_thread.join(3)
        assert not stream.play_thread.is_alive()
        assert engine.deliveries == [
            ("Inside.", "Whisper", "en"), ("Tail.", "Whisper", "en")]
        assert engine.instructions == "baseline"
    finally:
        engine.release.set()
        stream.close()


def test_alignment_source_contains_only_spoken_text(stream_factory):
    stream = stream_factory(InstructionEngine()).add_instruction("whispered", "Whisper")
    source = []
    class CaptureSource:
        def append_source(self, text):
            source.append(text)
    stream.alignment = CaptureSource()
    actions = list(stream._iter_inline_actions(["Hello.[whis", "pered]Secret."]))
    assert "".join(source) == "Hello.Secret."
    assert ("instruction", "Whisper") in actions
    stream.alignment = None


@pytest.mark.parametrize("generator", [False, True])
def test_unsupported_engine_fails_before_controlled_text_and_releases_lock(stream_factory, generator):
    engine = _PCMEngine()
    engine.can_consume_generators = generator
    stream = stream_factory(engine).add_instruction("whispered", "Whisper")
    with pytest.raises(NotImplementedError, match="inline instructions"):
        stream.feed("[whispered]Secret.").play()
    assert engine.queue.empty()
    assert not stream.play_lock.locked()


@pytest.mark.parametrize("generator", [False, True])
def test_mutating_setter_failure_restores_baseline(stream_factory, generator):
    class FailingSetter(InstructionEngine):
        def set_instructions(self, value):
            self.instructions = value
            if value == "Whisper":
                raise RuntimeError("setter failed")
    engine = FailingSetter()
    engine.can_consume_generators = generator
    stream = stream_factory(engine).add_instruction("whispered", "Whisper")
    with pytest.raises(RuntimeError, match="setter failed"):
        stream.feed("[whispered]Secret.").play()
    assert engine.instructions == "baseline"
    assert not engine.deliveries
    assert not stream.play_lock.locked()


def test_generator_synthesis_failure_restores_instruction(stream_factory):
    class FailingSynthesis(InstructionEngine):
        def synthesize(self, text, sentence_count=0):
            raise RuntimeError("synthesis failed")
    engine = FailingSynthesis()
    engine.can_consume_generators = True
    stream = stream_factory(engine).add_instruction("whispered", "Whisper")
    with pytest.raises(RuntimeError, match="synthesis failed"):
        stream.feed("[whispered]Secret.").play()
    assert engine.instructions == "baseline"
    assert not stream.play_lock.locked()


@pytest.mark.parametrize("generator", [False, True])
def test_cancellation_restores_instruction(stream_factory, generator):
    class Blocking(InstructionEngine):
        def post_init(self):
            super().post_init()
            self.started = threading.Event()
        def synthesize(self, text, sentence_count=0):
            if not isinstance(text, str):
                text = "".join(text)
            self.started.set()
            assert self.stop_synthesis_event.wait(3), "cancellation never reached engine"
            return False
    engine = Blocking()
    engine.can_consume_generators = generator
    stream = stream_factory(engine).add_instruction("whispered", "Whisper")
    stream.feed("[whispered]Secret.").play_async()
    assert engine.started.wait(2)
    stream.stop()
    assert engine.instructions == "baseline"
    assert not stream.play_lock.locked()


def test_fallback_inherits_instruction_and_restores_both_engines(stream_factory):
    class Failed(InstructionEngine):
        def synthesize(self, text, sentence_count=0):
            return False
    first, second = Failed(), InstructionEngine()
    second.instructions = "second baseline"
    stream = stream_factory([first, second]).add_instruction("whispered", "Whisper")
    stream.feed("[whispered]Secret.").play()
    assert second.deliveries == [("Secret.", "Whisper", "en")]
    assert first.instructions == "baseline"
    assert second.instructions == "second baseline"


def test_router_filters_instruction_for_cpu_route_and_restores_logical_baseline(stream_factory):
    from RealtimeTTS.engines.language_routing_engine import LanguageRoutingEngine
    from tests.test_language_routing_engine import FakeEngine
    english = FakeEngine("breeze", chunks=[b"\x00\x00" * 2400], direct_voice_control=True)
    spanish = FakeEngine("qwen", chunks=[b"\x00\x00" * 2400])
    router = LanguageRoutingEngine({"en": english, "es": spanish})
    router.set_instructions("baseline")
    stream = stream_factory(router, language_tags=True).add_instruction("surprised", "Excited")
    stream.feed("[surprised]Wow!<es>Hola.</es>Great.").play()
    assert ("instructions", "Excited") in english.calls
    assert not any(call[0] == "instructions" for call in spanish.calls)
    assert router.get_instructions() == "baseline"
