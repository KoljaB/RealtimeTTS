"""Pure tests for incremental text delivery into the stream iterators."""

from __future__ import annotations

import threading

from RealtimeTTS.text_to_stream import TextToAudioStream
from RealtimeTTS.threadsafe_generators import (
    AccumulatingThreadSafeGenerator,
    CharIterator,
)


def _stream_with_callbacks(events: list[str], characters: list[str]) -> TextToAudioStream:
    """Build only the text side of TextToAudioStream, without an audio engine."""

    stream = object.__new__(TextToAudioStream)
    stream.log_characters = False
    stream.generated_text = ""
    stream.on_text_stream_start = lambda: events.append("start")
    stream.on_text_stream_stop = lambda: events.append("stop")
    stream.on_character = characters.append
    stream._create_iterators()
    return stream


def test_feed_handles_strings_and_an_iterator_arriving_incrementally():
    events: list[str] = []
    characters: list[str] = []
    stream = _stream_with_callbacks(events, characters)

    iterator_started = threading.Event()
    release_iterator = threading.Event()

    def incoming_chunks():
        yield "from "
        iterator_started.set()
        assert release_iterator.wait(timeout=2)
        yield "iterator"

    stream.feed("hello ")
    stream.feed(incoming_chunks())
    source_iterator = stream.char_iter
    consumed: list[str] = []
    errors: list[BaseException] = []

    def consume():
        try:
            consumed.extend(source_iterator)
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)

    consumer = threading.Thread(target=consume)
    consumer.start()
    assert iterator_started.wait(timeout=2)

    # This string arrives while the iterator is paused between its chunks.
    assert stream.feed("!") is stream
    release_iterator.set()
    consumer.join(timeout=2)

    assert not consumer.is_alive()
    assert errors == []
    assert "".join(consumed) == "hello from iterator!"
    assert characters == consumed
    assert stream.text() == "hello from iterator!"
    assert events == ["start", "stop"]
    # Exhaustion resets the stream's input iterator for the next turn.
    assert stream.char_iter.items == []

    stream.feed("again")
    assert "".join(stream.char_iter) == "again"
    assert stream.text() == "hello from iterator!again"
    assert events == ["start", "stop", "start", "stop"]


def test_accumulating_generator_reports_callbacks_and_exhaustion():
    events: list[str] = []
    source = iter(["first", " second"])
    wrapped = AccumulatingThreadSafeGenerator(
        source,
        on_first_text_chunk=lambda: events.append("start"),
        on_last_text_chunk=lambda: events.append("stop"),
    )

    assert list(wrapped) == ["first", " second"]
    assert wrapped.is_exhausted()
    assert wrapped.accumulated_text() == "first second"
    assert events == ["start", "stop"]


def test_char_iterator_stop_stops_without_marking_input_as_exhausted():
    characters: list[str] = []
    last_callbacks: list[str] = []
    iterator: CharIterator

    def on_character(character: str):
        characters.append(character)
        if len(characters) == 3:
            iterator.stop()

    def endless_chunks():
        while True:
            yield "x"

    iterator = CharIterator(
        on_character=on_character,
        on_last_text_chunk=lambda: last_callbacks.append("stop"),
    )
    iterator.add(endless_chunks())

    assert list(iterator) == ["x", "x", "x"]
    assert iterator.immediate_stop.is_set()
    assert iterator.iterated_text == "xxx"
    # An immediate stop is distinct from normal iterator exhaustion.
    assert last_callbacks == []
