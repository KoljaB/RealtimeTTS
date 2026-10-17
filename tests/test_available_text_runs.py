"""Public playback tests with the real stream2sentence 1.1+ splitter."""

import threading
from itertools import groupby

import pytest

from RealtimeTTS import TextToAudioStream
from RealtimeTTS.inline_markup import InlineMarkupParser
from RealtimeTTS.threadsafe_generators import CharIterator, AccumulatingThreadSafeGenerator
import RealtimeTTS.stream_player as player_module
from tests.test_text_to_stream_alignment import _FakePyAudio, _PCMEngine


PARTS = [
    "All right, the first check is complete. ",
    "We can do that tomorrow, ",
    "though.",
]
TEXT = "".join(PARTS)
WHOLE = ["All right,", "the first check is complete.", "We can do that tomorrow, though."]
FRAGMENTS = WHOLE[:2] + ["We can do that tomorrow,", "though."]
PLAY_OPTIONS = dict(
    fast_sentence_fragment_allsentences=True,
    minimum_first_fragment_length=9,
    force_first_fragment_after_words=float("inf"),
)


class RecordingEngine(_PCMEngine):
    def post_init(self):
        super().post_init()
        self.spoken = []
        self.states = []
        self.language = "en"
        self.current_voice = "original"
        self.started = threading.Event()
        self.stream = None

    def get_language(self):
        return self.language

    def set_language(self, value):
        self.language = value

    def set_voice(self, value):
        self.current_voice = value

    def synthesize(self, text, sentence_count=0):
        if not isinstance(text, str):
            text = "".join(text)
        self.spoken.append(text)
        self.states.append((text, self.language, self.current_voice, self.stream._synthesis_speed))
        self.started.set()
        self.queue.put(b"\0" * 640)
        return True


class BlockingFirstEngine(RecordingEngine):
    def post_init(self):
        super().post_init()
        self.release = threading.Event()

    def synthesize(self, text, sentence_count=0):
        result = super().synthesize(text, sentence_count)
        if len(self.spoken) == 1:
            assert self.release.wait(3), "test did not release synthesis"
        return result

    def stop(self):
        self.release.set()


@pytest.fixture
def make_stream(monkeypatch):
    monkeypatch.setattr(player_module.pyaudio, "PyAudio", _FakePyAudio)
    streams = []

    def make(engine=None, **kwargs):
        engine = engine or RecordingEngine()
        stream = TextToAudioStream(engine, tokenizer="rule-based", **kwargs)
        engine.stream = stream
        streams.append(stream)
        return stream

    yield make
    for stream in streams:
        stream.close()


def join_playback(stream):
    thread = stream.play_thread
    assert thread is not None
    thread.join(4)
    assert not thread.is_alive(), "playback did not finish"
    assert not stream.error_flag


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("available", [False, True], ids=["legacy", "available"])
@pytest.mark.parametrize("input_kind", ["string", "queued_strings", "iterator_chunk"])
def test_public_playback_keeps_queued_sentences_and_character_callbacks(
    make_stream, asynchronous, available, input_kind
):
    characters, starts, stops, completed = [], [], [], []
    stream = make_stream(
        on_character=characters.append,
        on_text_stream_start=lambda: starts.append(True),
        on_text_stream_stop=lambda: stops.append(True),
    )
    if input_kind == "queued_strings":
        for part in PARTS:
            stream.feed(part)
    else:
        stream.feed(TEXT if input_kind == "string" else iter([TEXT]))
    play = stream.play_async if asynchronous else stream.play
    play(use_available_text_runs=available, on_sentence_synthesized=completed.append, **PLAY_OPTIONS)
    if asynchronous:
        join_playback(stream)
    assert not stream.error_flag
    assert stream.engine.spoken == (WHOLE if available else FRAGMENTS)
    assert completed == stream.engine.spoken
    assert characters == list(TEXT)
    assert starts == stops == [True]
    assert stream.text() == TEXT


@pytest.mark.parametrize("character_source", [False, True])
def test_explicit_every_fragment_and_character_sources_keep_quick_yielding(
    make_stream, character_source
):
    stream = make_stream()
    stream.feed(iter(TEXT) if character_source else TEXT).play(
        fast_sentence_fragment_allsentences_multiple=not character_source,
        **PLAY_OPTIONS,
    )
    assert not stream.error_flag
    assert stream.engine.spoken == FRAGMENTS


def test_snapshots_are_bounded_ordered_and_never_prefetch_iterators():
    pulls, characters = [], []

    def source():
        pulls.append(1)
        yield "iterator chunk"
        pulls.append(2)
        yield "future"

    chars = CharIterator(on_character=characters.append)
    chars.add("ab")
    chars.add("cd")
    chars.add(source())
    chars.add("last")
    assert chars.next_available(3) == "abc"
    assert chars.next_available(3) == "d"
    assert pulls == []
    assert chars.next_available() == "iterator chunk"
    assert pulls == [1]
    assert chars.next_available() == "future"
    assert pulls == [1, 2]
    assert chars.next_available() == "last"
    with pytest.raises(StopIteration):
        chars.next_available()
    assert "".join(characters) == chars.iterated_text == "abcditerator chunkfuturelast"


@pytest.mark.parametrize("feed_at", [1, 3])
def test_snapshot_cap_and_callback_feeds_do_not_grow_the_current_snapshot(feed_at):
    characters = []
    chars = CharIterator()

    def observe(character):
        characters.append(character)
        if len(characters) == feed_at:
            chars.add("late")

    chars.on_character = observe
    chars.add("abcd")
    wrapped = AccumulatingThreadSafeGenerator(chars)
    chunks = wrapped.iter_available_chunks()
    assert next(chunks) == "abcd"
    assert next(chunks) == "late"
    with pytest.raises(StopIteration):
        next(chunks)
    assert wrapped.accumulated_text() == "".join(characters) == "abcdlate"
    assert wrapped.is_exhausted()

    large = CharIterator()
    large.add("\U0001f600" * 5000)
    assert len(large.next_available()) == 4096
    assert len(large.next_available()) == 904


def test_stop_from_character_callback_stops_the_snapshot():
    chars = CharIterator()
    observed = []

    def observe(character):
        observed.append(character)
        if len(observed) == 5:
            chars.stop()

    chars.on_character = observe
    chars.add("abcdefghijklmnopqrstuvwxyz")
    wrapped = AccumulatingThreadSafeGenerator(chars)
    assert list(wrapped.iter_available_chunks()) == ["abcde"]
    assert observed == list("abcde")
    assert wrapped.accumulated_text() == "abcde"


def test_first_audio_arrives_before_the_generator_produces_more_input(make_stream):
    release, audio = threading.Event(), threading.Event()

    def source():
        yield "All right, the first check is "
        assert release.wait(3), "test did not release input"
        yield "complete. We can do that tomorrow, though."

    stream = make_stream()
    try:
        stream.feed(source()).play_async(on_audio_chunk=lambda _: audio.set(), **PLAY_OPTIONS)
        assert audio.wait(2), "available-text buffering delayed the first audio"
        assert stream.engine.spoken[0] == "All right,"
    finally:
        release.set()
        join_playback(stream)


def _normalized(actions):
    result = []
    for is_text, group in groupby(actions, lambda item: item[0] == "text"):
        if is_text:
            result.append(("text", "".join(value for _, value in group)))
        else:
            result.extend(group)
    return result


def test_coalescing_preserves_controls_and_partial_tags_at_every_split():
    source = "A<es slow>B</es>C[speed=2]D[voice]E[pause]F[unknown]G<incomplete"

    def parser():
        return InlineMarkupParser(
            1.0, speed_tags=True, language_tags=True,
            voice_tags={"voice": "alternate"}, pause_tags={"pause": 0.1},
        )

    expected = _normalized(parser().iter_actions([source]))
    for split in range(len(source) + 1):
        actual = parser().iter_actions(
            [source[:split], source[split:]], coalesce_text=True
        )
        assert _normalized(actual) == expected


def test_real_splitter_does_not_merge_across_inline_controls(make_stream):
    stream = make_stream(language_tags=True, speed_tags=True)
    stream.add_voice("voice", "alternate").add_pause("pause", 0.01)
    stream.feed("Before.<es>Hola.</es>[voice]Voice.[speed=2]Fast.[pause]After.").play(
        minimum_sentence_length=1,
        minimum_first_fragment_length=1,
        fast_sentence_fragment=False,
    )
    assert not stream.error_flag
    assert stream.engine.states == [
        ("Before.", "en", "original", 1.0),
        ("Hola.", "es", "original", 1.0),
        ("Voice.", "en", "alternate", 1.0),
        ("Fast.", "en", "alternate", 2.0),
        ("After.", "en", "alternate", 2.0),
    ]


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("feed_count", [1, 2])
def test_late_feed_preserves_available_and_fragment_options(make_stream, available, feed_count):
    ended = threading.Event()
    engine = BlockingFirstEngine()
    stream = make_stream(engine, on_text_stream_stop=ended.set)
    try:
        stream.feed("Start.").play_async(use_available_text_runs=available, **PLAY_OPTIONS)
        assert ended.wait(2) and engine.started.wait(2)
        if feed_count == 1:
            stream.feed(TEXT)
        else:
            stream.feed(PARTS[0])
            stream.feed("".join(PARTS[1:]))
        engine.release.set()
        join_playback(stream)
        assert engine.spoken == ["Start."] + (WHOLE if available else FRAGMENTS)
    finally:
        engine.release.set()


@pytest.mark.parametrize("available", [False, True])
def test_late_feed_preserves_custom_tokenizer_callbacks_context_and_pauses(
    make_stream, monkeypatch, available
):
    import stream2sentence
    import RealtimeTTS.text_to_stream as text_module

    ended = threading.Event()
    engine = BlockingFirstEngine()
    stream = make_stream(engine, on_text_stream_stop=ended.set)
    options_seen, before, completed, pauses = [], [], [], []
    tokenize = lambda text: text.split("|")

    class RecordingSplitter:
        @staticmethod
        def generate_sentences(source, **options):
            options_seen.append(options)
            yield from stream2sentence.generate_sentences(source, **options)

    enqueue_silence = stream._enqueue_silence

    def record_pause(duration):
        if duration:
            pauses.append(duration)
        return enqueue_silence(duration)

    monkeypatch.setattr(text_module, "_get_stream2sentence", lambda: RecordingSplitter)
    monkeypatch.setattr(stream, "_enqueue_silence", record_pause)
    try:
        stream.feed("Start.").play_async(
            use_available_text_runs=available,
            fast_sentence_fragment=False,
            fast_sentence_fragment_allsentences=False,
            tokenize_sentences=tokenize,
            minimum_sentence_length=1,
            minimum_first_fragment_length=1,
            context_size=2,
            context_size_look_overhead=3,
            comma_silence_duration=0.01,
            sentence_silence_duration=0.02,
            default_silence_duration=0.03,
            before_sentence_synthesized=before.append,
            on_sentence_synthesized=completed.append,
        )
        assert ended.wait(2) and engine.started.wait(2)
        stream.feed("Alpha|Beta")
        engine.release.set()
        join_playback(stream)
        assert before == completed == engine.spoken == ["Start.", "Alpha", "Beta"]
        assert pauses == pytest.approx([0.02, 0.03, 0.03])
        assert len(options_seen) == 2
        assert all(options["tokenize_sentences"] is tokenize for options in options_seen)
        assert all(options["context_size_look_overhead"] == 3 for options in options_seen)
    finally:
        engine.release.set()


@pytest.mark.parametrize("source", ["", [], [""]])
def test_exhausted_empty_input_does_not_restart_playback(make_stream, monkeypatch, source):
    stream = make_stream()
    calls = []
    play = stream.play

    def record_play(*args, **kwargs):
        calls.append(kwargs)
        return play(*args, **kwargs)

    monkeypatch.setattr(stream, "play", record_play)
    stream.feed(source if isinstance(source, str) else iter(source)).play()
    assert len(calls) == 1
    assert not stream.error_flag
    assert stream.engine.spoken == []
    stream.feed("Fresh speech.").play(**PLAY_OPTIONS)
    assert stream.engine.spoken == ["Fresh speech."]


def test_stop_discards_queued_runs_and_next_play_is_fresh(make_stream):
    engine = BlockingFirstEngine()
    stream = make_stream(engine)
    try:
        stream.feed(TEXT * 4).play_async(**PLAY_OPTIONS)
        assert engine.started.wait(2)
        stream.stop()
        assert not stream.is_playing()
        assert engine.spoken == ["All right,"]
        stream.feed("Fresh speech.").play(**PLAY_OPTIONS)
        assert not stream.error_flag
        assert engine.spoken == ["All right,", "Fresh speech."]
    finally:
        engine.release.set()
