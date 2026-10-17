from __future__ import annotations

import threading
import wave

import pytest

from RealtimeTTS import QwenEngine, QwenVoice, TextToAudioStream
from RealtimeTTS.engines.qwen_server_engine import QwenServerEngine
from RealtimeTTS.inline_markup import InlineMarkupParser
import RealtimeTTS.stream_player as player_module
import RealtimeTTS.text_to_stream as text_module
from tests.test_text_to_stream_alignment import _FakePyAudio, _PCMEngine


class WholeRunSplitter:
    @staticmethod
    def generate_sentences(source, **options):
        value = "".join(getattr(chunk, "text", chunk) for chunk in source)
        if value:
            yield value


class ObservingEngine(_PCMEngine):
    def post_init(self):
        super().post_init()
        self.language = "en"
        self.current_voice = "original"
        self.language_calls = []
        self.voice_calls = []
        self.observations = []
        self.stream = None

    def set_language(self, language):
        self.language_calls.append(language)
        self.language = language

    def get_language(self):
        return self.language

    def set_voice(self, voice):
        self.voice_calls.append(voice)
        self.current_voice = voice

    def synthesize(self, text, sentence_count=0):
        if not isinstance(text, str):
            text = "".join(text)
        speed = self.stream._synthesis_speed if self.stream is not None else None
        self.observations.append(
            (text, self.language, self.current_voice, speed)
        )
        return super().synthesize(text, sentence_count)


@pytest.fixture
def stream_factory(monkeypatch):
    monkeypatch.setattr(player_module.pyaudio, "PyAudio", _FakePyAudio)
    monkeypatch.setattr(text_module, "_get_stream2sentence", lambda: WholeRunSplitter)

    def make(engine=None, **options):
        value = engine if engine is not None else ObservingEngine()
        stream = TextToAudioStream(
            value,
            tokenizer="rule-based",
            playout_chunk_size=640,
            **options,
        )
        if isinstance(value, ObservingEngine):
            value.stream = stream
        return stream

    return make


def _coalesce(actions):
    output = []
    for kind, value in actions:
        if kind == "text" and output and output[-1][0] == "text":
            output[-1] = ("text", output[-1][1] + value)
        else:
            output.append((kind, value))
    return output


def _observations(engine):
    return [
        (text.strip(), language, voice, speed)
        for text, language, voice, speed in engine.observations
        if text.strip()
    ]


def test_language_parser_survives_every_split_position_and_canonical_form():
    cases = {
        "A<es>Hola</es>Z": [
            ("text", "A"),
            ("language", "es"),
            ("text", "Hola"),
            ("language", "en"),
            ("text", "Z"),
        ],
        "A<es slow>Hola</es>Z": [
            ("text", "A"),
            ("language", "es"),
            ("speed", 0.75),
            ("text", "Hola"),
            ("language", "en"),
            ("speed", 1.25),
            ("text", "Z"),
        ],
        'A<lang code="es">Hola</lang>Z': [
            ("text", "A"),
            ("language", "es"),
            ("text", "Hola"),
            ("language", "en"),
            ("text", "Z"),
        ],
    }

    for source, expected in cases.items():
        for split in range(len(source) + 1):
            parser = InlineMarkupParser(
                1.25,
                language_tags=True,
                default_language="en",
            )
            actual = _coalesce(
                list(parser.iter_actions((source[:split], source[split:])))
            )
            assert actual == expected
            assert all(
                isinstance(value, float)
                for kind, value in actual
                if kind == "speed"
            )


def test_language_scope_embeds_speed_without_speed_tags():
    parser = InlineMarkupParser(
        1.0,
        language_tags=True,
        default_language="en",
    )
    actual = _coalesce(
        list(parser.iter_actions(["A<es speed=0.8>B</es>C"]))
    )
    assert actual == [
        ("text", "A"),
        ("language", "es"),
        ("speed", 0.8),
        ("text", "B"),
        ("language", "en"),
        ("speed", 1.0),
        ("text", "C"),
    ]


@pytest.mark.parametrize("code", ["en", "de", "es", "pt"])
@pytest.mark.parametrize(("modifier", "expected_speed"), [
    ("slow", 0.75),
    ("fast", 1.2),
])
def test_compact_languages_support_slow_and_fast(code, modifier, expected_speed):
    parser = InlineMarkupParser(
        1.0,
        language_tags=True,
        default_language="en",
    )
    source = f"<{code} {modifier}>word</{code}>"
    assert _coalesce(list(parser.iter_actions([source]))) == [
        ("language", code),
        ("speed", expected_speed),
        ("text", "word"),
        ("language", "en"),
        ("speed", 1.0),
    ]


@pytest.mark.parametrize("source", ["<es>word</es>", '<lang code="es">word</lang>'])
def test_language_tags_are_literal_when_disabled(source):
    parser = InlineMarkupParser(
        1.0,
        language_tags=False,
        default_language="en",
    )
    assert _coalesce(list(parser.iter_actions([source]))) == [("text", source)]


@pytest.mark.parametrize(
    "source",
    [
        "x<es speed=oops>y",
        "x<es medium>y",
        "x<lang>y</lang>",
        "x<lang code=>y",
        "x<zz>y</zz>",
    ],
)
def test_malformed_or_unknown_language_markup_remains_literal(source):
    parser = InlineMarkupParser(
        1.0,
        language_tags=True,
        default_language="en",
    )
    assert _coalesce(list(parser.iter_actions([source]))) == [("text", source)]

    disabled = InlineMarkupParser(
        1.0,
        language_tags=False,
        default_language="en",
    )
    assert _coalesce(list(disabled.iter_actions([source]))) == [("text", source)]


def test_language_parser_releases_body_text_before_waiting_for_closing_tag():
    release = threading.Event()

    def source():
        yield "A<es>Body"
        release.wait(2)
        yield "</es>Z"

    parser = InlineMarkupParser(
        1.0,
        language_tags=True,
        default_language="en",
    )
    iterator = parser.iter_actions(source())
    observed = []
    text_seen = ""
    try:
        for _ in range(64):
            item = next(iterator)
            observed.append(item)
            if item[0] == "text":
                text_seen += item[1]
            if text_seen.endswith("Body"):
                break
        else:
            pytest.fail("parser did not release body text before the close tag")
        assert text_seen.endswith("Body")
        assert not release.is_set()
    finally:
        release.set()
        iterator.close()

    assert _coalesce(observed) == [
        ("text", "A"),
        ("language", "es"),
        ("text", "Body"),
    ]


def test_nested_language_speed_scopes_restore_voice_language_and_tempo(stream_factory):
    engine = ObservingEngine()
    stream = stream_factory(
        engine,
        speed=1.25,
        language="en",
        language_tags=True,
        language_voices={"es": "es-voice", "de": "de-voice"},
    )
    stream.feed("A<es slow>B<de fast>C</de>D</es>E").play()

    assert _observations(engine) == [
        ("A", "en", "original", 1.25),
        ("B", "es", "es-voice", 0.75),
        ("C", "de", "de-voice", 1.2),
        ("D", "es", "es-voice", 0.75),
        ("E", "en", "original", 1.25),
    ]
    assert engine.language == "en"
    assert engine.current_voice == "original"
    assert engine.voice_calls == [
        "es-voice",
        "de-voice",
        "es-voice",
        "original",
    ]


def test_sentence_splitter_receives_active_language_for_each_scope(
    stream_factory, monkeypatch
):
    calls = []

    class RecordingSplitter:
        @staticmethod
        def generate_sentences(source, **options):
            value = "".join(getattr(chunk, "text", chunk) for chunk in source)
            calls.append((value.strip(), options["language"]))
            if value:
                yield value

    monkeypatch.setattr(text_module, "_get_stream2sentence", lambda: RecordingSplitter)
    engine = ObservingEngine()
    stream = stream_factory(
        engine,
        language="en",
        language_tags=True,
    )
    stream.feed("One. <es>Dos.</es> Three.").play()

    assert calls == [
        ("One.", "en"),
        ("Dos.", "es"),
        ("Three.", "en"),
    ]
    assert [
        (text, language)
        for text, language, _voice, _speed in _observations(engine)
    ] == [
        ("One.", "en"),
        ("Dos.", "es"),
        ("Three.", "en"),
    ]


def test_language_tempo_reaches_callbacks_and_wav(stream_factory, tmp_path):
    engine = ObservingEngine()
    stream = stream_factory(
        engine,
        language="en",
        language_tags=True,
    )
    chunks = []
    output = tmp_path / "language-tempo.wav"
    stream.feed("One. <es slow>Dos.</es>Three.").play(
        output_wavfile=str(output),
        on_audio_chunk=chunks.append,
        muted=True,
    )

    assert chunks
    closed = [segment for segment in stream.tempo_timeline.segments if segment.closed]
    assert [segment.speed for segment in closed] == pytest.approx(
        [1.0, 0.75, 1.0]
    )
    assert stream.tempo_timeline.output_duration == pytest.approx(
        1.0 + (1.0 / 0.75) + 1.0,
        abs=0.15,
    )
    with wave.open(str(output), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getsampwidth() == 2
        pcm = wav.readframes(wav.getnframes())
    assert pcm == b"".join(chunks)
    assert len(pcm) > 0


def test_generator_engine_applies_language_scope(stream_factory):
    engine = ObservingEngine()
    engine.can_consume_generators = True
    stream = stream_factory(
        engine,
        language="en",
        language_tags=True,
        language_voices={"es": "es-voice"},
    )
    stream.feed("A<es>Hola</es>Z").play()

    assert _observations(engine) == [
        ("A", "en", "original", 1.0),
        ("Hola", "es", "es-voice", 1.0),
        ("Z", "en", "original", 1.0),
    ]
    assert engine.language == "en"
    assert engine.current_voice == "original"


def test_language_scope_failure_restores_engine_and_stream_baseline(stream_factory):
    class FailingGeneratorEngine(ObservingEngine):
        def post_init(self):
            super().post_init()
            self.can_consume_generators = True

        def synthesize(self, text, sentence_count=0):
            if not isinstance(text, str):
                text = "".join(text)
            if text.strip() == "Inside":
                raise RuntimeError("synthetic language failure")
            return super().synthesize(text, sentence_count)

    engine = FailingGeneratorEngine()
    stream = stream_factory(
        engine,
        speed=1.25,
        language="en",
        language_tags=True,
        language_voices={"es": "es-voice"},
    )
    with pytest.raises(RuntimeError, match="synthetic language failure"):
        stream.feed("Before<es slow>Inside").play()

    assert engine.language == "en"
    assert engine.current_voice == "original"
    assert stream._synthesis_speed == pytest.approx(1.25)
    assert stream.player.speed == pytest.approx(1.25)
    assert not stream.is_playing()


def test_language_scope_cancellation_restores_engine_baseline(stream_factory):
    class BlockingGeneratorEngine(ObservingEngine):
        def post_init(self):
            super().post_init()
            self.can_consume_generators = True
            self.release = threading.Event()
            self.started = threading.Event()

        def synthesize(self, text, sentence_count=0):
            if not isinstance(text, str):
                text = "".join(text)
            if text.strip() == "Inside":
                self.started.set()
                while (
                    not self.stop_synthesis_event.is_set()
                    and not self.release.wait(0.01)
                ):
                    pass
                return False
            return super().synthesize(text, sentence_count)

    engine = BlockingGeneratorEngine()
    stream = stream_factory(
        engine,
        language="en",
        language_tags=True,
        language_voices={"es": "es-voice"},
    )
    stream.feed("Before<es>Inside").play_async()
    assert engine.started.wait(2)
    stream.stop()
    engine.release.set()

    assert not stream.is_playing()
    assert engine.language == "en"
    assert engine.current_voice == "original"
    assert stream._synthesis_speed == pytest.approx(1.0)


class UnsupportedLanguageEngine(_PCMEngine):
    def post_init(self):
        super().post_init()
        self.language = "en"
        self.can_consume_generators = True
        self.spoken = []

    def synthesize(self, text, sentence_count=0):
        if not isinstance(text, str):
            text = "".join(text)
        self.spoken.append(text)
        return super().synthesize(text, sentence_count)


def test_base_engine_rejects_language_scope_before_tagged_speech(stream_factory):
    engine = UnsupportedLanguageEngine()
    stream = stream_factory(
        engine,
        language="en",
        language_tags=True,
    )
    with pytest.raises(NotImplementedError, match="inline language"):
        stream.feed("<es>Hola</es>").play()
    assert engine.spoken == []
    assert engine.language == "en"


@pytest.mark.parametrize("actual_language", ["english", "auto"])
def test_constructor_uses_tokenizer_language_and_restores_engine_language(
    stream_factory, actual_language
):
    engine = ObservingEngine()
    engine.language = actual_language
    stream = stream_factory(engine, language_tags=True)

    assert stream.language == "en"
    stream.feed("Before<es>Hola</es>After").play()

    assert [
        (text, language)
        for text, language, _voice, _speed in _observations(engine)
    ] == [
        ("Before", actual_language),
        ("Hola", "es"),
        ("After", actual_language),
    ]
    assert engine.language == actual_language


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("es", "spanish"), ("Auto", "auto")],
)
def test_native_qwen_language_setter_does_not_warm_up(requested, expected):
    engine = object.__new__(QwenEngine)
    original_voice = QwenVoice("voice", language="english", speaker="speaker")
    engine.current_voice = original_voice
    engine._synthesis_lock = threading.RLock()
    warmup_calls = []
    engine.warmup = lambda: warmup_calls.append(True)

    engine.set_language(requested)

    assert engine.get_language() == expected
    assert engine.current_voice is not original_voice
    assert original_voice.language == "english"
    assert warmup_calls == []


def test_native_qwen_inline_voice_selection_uses_no_warmup_hook():
    engine = object.__new__(QwenEngine)
    engine._synthesis_lock = threading.RLock()
    engine.current_voice = QwenVoice(
        "original",
        language="english",
        speaker="speaker",
    )
    calls = []

    def set_voice(voice, *, warmup=True):
        calls.append((voice, warmup))

    engine.set_voice = set_voice
    mapped_voice = QwenVoice(
        "mapped",
        language="spanish",
        speaker="speaker",
    )

    engine.set_inline_voice(mapped_voice)

    assert calls == [(mapped_voice, False)]


def test_qwen_server_language_payload_is_local_and_has_no_http_call(monkeypatch):
    class NoNetworkSession:
        def __init__(self):
            self.calls = 0

        def close(self):
            pass

        def post(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("language payload unexpectedly performed HTTP")

        def get(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("language payload unexpectedly performed HTTP")

    session = NoNetworkSession()
    monkeypatch.setattr(
        "RealtimeTTS.engines.qwen_server_engine.requests.Session",
        lambda: session,
    )
    engine = QwenServerEngine(
        base_url="http://127.0.0.1:1",
        voice="alex",
        language="en",
    )
    try:
        engine.set_language("es")
        assert engine.get_language() == "es"
        assert engine._payload("hola")["language"] == "es"
        assert session.calls == 0
    finally:
        engine.shutdown()
