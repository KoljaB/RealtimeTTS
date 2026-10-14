import pytest

import RealtimeTTS.stream_player as player_module
from tests.test_inline_language import stream_factory


def test_local_language_switch_attributes_outer_period_and_no_double_gap(stream_factory, monkeypatch):
    stream = stream_factory(language_tags=True)
    gaps = []
    monkeypatch.setattr(stream, "_enqueue_silence", lambda value: (gaps.append(value) if value else None) or value)
    stream.feed("Tree is <es>el árbol</es>. Try saying it.").play(
        muted=True, comma_silence_duration=.15, sentence_silence_duration=.30,
    )
    assert [text.strip() for text, *_ in stream.engine.observations] == [
        "Tree is", "el árbol", "Try saying it.",
    ]
    assert gaps == [.08, .30, .30]


def test_local_sentence_pause_covers_transition(stream_factory, monkeypatch):
    stream = stream_factory(language_tags=True)
    gaps = []
    monkeypatch.setattr(stream, "_enqueue_silence", lambda value: (gaps.append(value) if value else None) or value)
    stream.feed("<es>Hola.</es>Again.").play(muted=True, sentence_silence_duration=.30)
    assert gaps == [.30, .30]


def test_local_transition_gap_can_be_disabled(stream_factory, monkeypatch):
    stream = stream_factory(language_tags=True)
    gaps = []
    monkeypatch.setattr(stream, "_enqueue_silence", lambda value: (gaps.append(value) if value else None) or value)
    stream.feed("Tree <es>árbol</es> tree").play(muted=True, language_transition_silence_duration=0)
    assert not gaps


def test_custom_format_does_not_consume_boundary_pause(stream_factory, monkeypatch):
    stream = stream_factory(language_tags=True)
    monkeypatch.setattr(
        stream,
        "_output_stream_info",
        lambda: (player_module.pyaudio.paCustomFormat, 1, 24000),
    )
    calls = []
    original = stream._enqueue_silence

    def capture(duration):
        emitted = original(duration)
        calls.append((duration, emitted))
        return emitted

    monkeypatch.setattr(stream, "_enqueue_silence", capture)
    stream.feed("One.<es>Hola").play(
        muted=True,
        sentence_silence_duration=0.30,
    )

    assert [requested for requested, _ in calls] == pytest.approx([0.0, 0.30, 0.08, 0.0])
    assert [emitted for _, emitted in calls] == [0.0, 0.0, 0.0, 0.0]
