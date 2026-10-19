from __future__ import annotations

import threading

import pytest

from RealtimeTTS.alignment import FakeWordAligner, WordAlignment
from RealtimeTTS.inline_markup import InlineMarkupParser, Vocalization
from tests.test_inline_instructions import InstructionEngine
from tests.test_inline_language import _coalesce, stream_factory


class VocalEngine(InstructionEngine):
    def post_init(self):
        super().post_init()
        self.instruction_control = True
        self.vocalizations = {"sigh": "(sigh)", "laugh": "(laugh)",
                              "clear_throat": "(clear throat)"}
        self.instruction_calls = []

    def set_instructions(self, value):
        self.instruction_calls.append(value)
        super().set_instructions(value)


def test_parser_keeps_vocalizations_distinct_at_every_chunk_boundary():
    source = "A[whispered]B[sigh][pause:0.1]<laugh/>C[unknown](aside)"
    expected = [
        ("text", "A"), ("instruction", "Whisper"), ("text", "B"),
        ("vocalization", Vocalization("sigh", "Gentle sigh")),
        ("pause", ("pause", 0.1)),
        ("vocalization", Vocalization("laugh")),
        ("text", "C[unknown](aside)"),
    ]
    for split in range(len(source) + 1):
        parser = InlineMarkupParser(
            1.0, instruction_tags={"whispered": "Whisper"},
            vocalization_tags={"sigh": "Gentle sigh", "laugh": None},
            pause_tags={"pause": 0.7},
        )
        assert _coalesce(parser.iter_actions((source[:split], source[split:]))) == expected


@pytest.mark.parametrize("tags", [
    {"en": None}, {"pause": None}, {"speed": None}, {"bad name": None},
    {"": None}, {"sigh": None, "SIGH": None},
])
def test_invalid_vocalization_names_are_rejected(tags):
    with pytest.raises(ValueError):
        InlineMarkupParser(1.0, vocalization_tags=tags)


def test_registrations_reject_ambiguous_meanings_and_invalid_directions(stream_factory):
    with stream_factory() as stream:
        stream.add_instruction("sigh", "Sticky")
        with pytest.raises(ValueError, match="conflicting"):
            stream.add_vocalization("sigh")
    with stream_factory() as stream:
        stream.add_vocalization("sigh")
        with pytest.raises(ValueError, match="conflicting"):
            stream.add_instruction("sigh", "Sticky")
        with pytest.raises(TypeError, match="string"):
            stream.add_vocalization("laugh", instructions=123)
        stream.add_pause("breath", 0.1)
        with pytest.raises(ValueError, match="conflicting"):
            stream.add_vocalization("breath")


@pytest.mark.parametrize("generator", [False, True])
@pytest.mark.parametrize("available", [False, True])
def test_one_shot_restores_delivery_language_voice_and_speed(stream_factory, generator, available):
    engine = VocalEngine()
    engine.can_consume_generators = generator
    with stream_factory(engine, language_tags=True) as stream:
        stream.add_instruction("whispered", "Whisper")
        stream.add_vocalization("sigh", instructions="Gentle sigh")
        stream.add_vocalization("laugh")
        stream.add_voice("other", "second voice")
        stream.feed(
            "<en speed=1.1>[other][whispered]Before[sigh]After"
            "<es fast>Hola[laugh]Adios</es>End</en>"
        ).play(use_available_text_runs=available, default_silence_duration=0)
        assert engine.deliveries == [
            ("Before", "Whisper", "en"), ("(sigh)", "Gentle sigh", "en"),
            ("After", "Whisper", "en"), ("Hola", "Whisper", "es"),
            ("(laugh)", "Whisper", "es"), ("Adios", "Whisper", "es"),
            ("End", "Whisper", "en"),
        ]
        assert [row[3] for row in engine.observations] == [1.1, 1.1, 1.1, 1.2, 1.2, 1.2, 1.1]
        assert all(row[2] == "second voice" for row in engine.observations)
        assert engine.instructions == "baseline"
        assert engine.current_voice == "original"
        assert engine.language == "en"


@pytest.mark.parametrize("generator", [False, True])
def test_sound_only_input_and_explicit_pause_order(stream_factory, monkeypatch, generator):
    engine = VocalEngine()
    engine.can_consume_generators = generator
    events = []
    synthesize = engine.synthesize
    def record_sound(text, sentence_count=0):
        text = text if isinstance(text, str) else "".join(text)
        events.append(text)
        return synthesize(text, sentence_count)
    monkeypatch.setattr(engine, "synthesize", record_sound)
    with stream_factory(engine) as stream:
        stream.add_vocalization("sigh", instructions="Gentle sigh")
        stream.add_vocalization("clear_throat")
        stream.add_pause("pause")
        monkeypatch.setattr(stream, "_enqueue_silence",
                            lambda value: events.append(value) or value)
        stream.feed(iter(["[pause:0.07][si", "gh][pause:0.03][clear_throat]"])).play()
        assert events == [0.07, "(sigh)", 0.03, "(clear throat)"]
        assert engine.instructions == "baseline"


@pytest.mark.parametrize("generator", [False, True])
@pytest.mark.parametrize("failure", ["setter", "synthesis", "false"])
def test_vocal_failure_restores_previous_style_without_retry(stream_factory, generator, failure):
    class Failing(VocalEngine):
        def set_instructions(self, value):
            super().set_instructions(value)
            if failure == "setter" and value == "Gentle sigh":
                raise RuntimeError("setter failed")
        def synthesize(self, text, sentence_count=0):
            if failure == "synthesis":
                raise RuntimeError("synthesis failed")
            if failure == "false":
                return False
            return super().synthesize(text, sentence_count)
    engine = Failing()
    engine.can_consume_generators = generator
    with stream_factory(engine) as stream:
        stream.add_instruction("whispered", "Whisper")
        stream.add_vocalization("sigh", instructions="Gentle sigh")
        with pytest.raises(RuntimeError, match="failed"):
            stream.feed("[whispered][sigh]Never synthesized").play()
        assert engine.instruction_calls == ["Whisper", "Gentle sigh", "Whisper", "baseline"]
        assert not engine.deliveries
        assert not stream.play_lock.locked()


@pytest.mark.parametrize("generator", [False, True])
def test_cancellation_restores_style_and_skips_following_speech(stream_factory, generator):
    class Blocking(VocalEngine):
        def post_init(self):
            super().post_init()
            self.started = threading.Event()
        def synthesize(self, text, sentence_count=0):
            self.started.set()
            assert self.stop_synthesis_event.wait(3)
            return False
    engine = Blocking()
    engine.can_consume_generators = generator
    with stream_factory(engine) as stream:
        stream.add_instruction("whispered", "Whisper")
        stream.add_vocalization("sigh", instructions="Gentle sigh")
        stream.feed("[whispered][sigh]Never synthesized").play_async()
        assert engine.started.wait(2)
        stream.stop()
        assert engine.instruction_calls == ["Whisper", "Gentle sigh", "Whisper", "baseline"]
        assert not stream.play_lock.locked()
        assert not engine.deliveries


@pytest.mark.parametrize("generator", [False, True])
def test_unsupported_sound_fails_before_it_can_be_spoken(stream_factory, generator):
    engine = InstructionEngine()
    engine.can_consume_generators = generator
    with stream_factory(engine).add_vocalization("sigh") as stream:
        with pytest.raises(NotImplementedError, match="vocalization"):
            stream.feed("[sigh]Never synthesized").play()
        assert not engine.deliveries
        assert not stream.play_lock.locked()


@pytest.mark.parametrize("generator", [False, True])
def test_alignment_excludes_sound_but_counts_its_audio(stream_factory, generator):
    engine = VocalEngine()
    engine.can_consume_generators = generator
    aligner = FakeWordAligner([WordAlignment("word", 0.0, 0.1)])
    with stream_factory(
        engine, alignment={"source": "model", "mode": "export", "aligner": aligner}
    ).add_vocalization("sigh") as stream:
        stream.feed("word [sigh]word").play(default_silence_duration=0)
        snapshot = stream.alignment.snapshot()
        assert snapshot.text == "word word"
        assert [call[0].strip() for call in aligner.calls] == ["word", "word"]
        assert [(span.text_start, span.text_end) for span in snapshot.words] == [(0, 4), (5, 9)]
        assert [span.start_time for span in snapshot.words] == pytest.approx([0, 2])
        assert not snapshot.errors


def test_terminal_display_uses_original_tags_without_shifting_offsets(stream_factory):
    from rich.text import Text
    from RealtimeTTS.terminal_highlighting import TerminalHighlighter
    highlighter = object.__new__(TerminalHighlighter)
    highlighter.input_text = "Hello [sigh]world[laugh]!"
    with stream_factory(VocalEngine()) as stream:
        stream.add_vocalization("sigh").add_vocalization("laugh")
        highlighter._prepare_input(stream)
        assert highlighter._input_plain == "Hello world!"
        assert highlighter._with_instructions(Text("Hello world!")).plain == highlighter.input_text
        source = []
        class Capture:
            def append_source(self, value):
                source.append(value)
        stream.alignment = Capture()
        try:
            list(stream._iter_inline_actions(iter(highlighter.input_text)))
            assert "".join(source) == highlighter._input_plain
        finally:
            stream.alignment = None


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
def test_remote_adapter_sends_one_native_cue_and_restores_direction(
    monkeypatch, stream_factory, routed, persistent
):
    from RealtimeTTS.engines.remote_tts_engine import RemoteTTSEngine
    from tests.test_remote_tts_engine import _install, _success
    from RealtimeTTS.engines.language_routing_engine import LanguageRoutingEngine
    wires = _install(monkeypatch, [_success] * 3, direct=True,
                     vocal_tags=["sigh", "clear throat"])
    engine = RemoteTTSEngine("http://provider.test", "mira", language="en")
    assert engine.get_vocalization_text("clear_throat") == "(clear throat)"
    provider = engine
    if routed:
        engine = LanguageRoutingEngine({"en": provider})
    baseline = engine.get_instructions()
    context = engine.create_stream_context() if persistent else None
    if context is not None:
        engine.set_stream_context(context)
    try:
        with stream_factory(engine) as stream:
            stream.add_instruction("whispered", "Whisper")
            stream.add_vocalization("sigh", instructions="Gentle sigh")
            stream.feed("[whispered]Before[sigh]After").play(default_silence_duration=0)
        assert len(wires) == (1 if persistent else 3)
        events = [event for wire in wires for event in wire.sent]
        texts = [event for event in events if event["type"] == "text"]
        assert [event["text"] for event in texts] == ["Before", "(sigh)", "After"]
        assert [event["instructions"] for event in texts] == ["Whisper", "Gentle sigh", "Whisper"]
        assert len([event for event in events if event["type"] == "flush"]) == 3
        assert engine.get_instructions() == baseline
    finally:
        if context is not None:
            engine.close_stream_context(context)
        engine.shutdown()
        if routed:
            provider.shutdown()


def test_router_checks_active_route_and_every_fallback():
    from RealtimeTTS.engines.language_routing_engine import LanguageRoutingEngine
    first, second, plain = VocalEngine(), VocalEngine(), InstructionEngine()
    for engine in (first, second, plain):
        engine.sample_rate = 24000
    router = LanguageRoutingEngine({"en": [first, second], "es": plain})
    try:
        assert router.get_vocalization_text("sigh", instructions="Gentle sigh") == "(sigh)"
        second.vocalizations["sigh"] = "<native-sigh>"
        with pytest.raises(NotImplementedError, match="different"):
            router.get_vocalization_text("sigh")
        second.vocalizations.clear()
        with pytest.raises(NotImplementedError, match="vocalization"):
            router.get_vocalization_text("sigh")
        router.set_language("es")
        with pytest.raises(NotImplementedError, match="vocalization"):
            router.get_vocalization_text("sigh")
    finally:
        router.shutdown()
