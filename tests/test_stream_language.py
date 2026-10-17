"""Language inheritance is exercised through real sentence splitting and playback."""

import pytest

import RealtimeTTS.stream_player as stream_player
from RealtimeTTS import TextToAudioStream
from tests.test_text_to_stream_alignment import _FakePyAudio, _PCMEngine


@pytest.fixture(autouse=True)
def fake_audio_device(monkeypatch):
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)


def speak(stream, text, asynchronous, **options):
    fragments = []
    stream.feed(text)
    play = stream.play_async if asynchronous else stream.play
    play(
        before_sentence_synthesized=fragments.append,
        minimum_sentence_length=1,
        minimum_first_fragment_length=1,
        **options,
    )
    if asynchronous:
        thread = stream.play_thread
        assert thread is not None
        thread.join(timeout=5)
        assert not thread.is_alive(), "play_async() did not finish"
    return fragments


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("de", "Wir besuchen Frl. Schmidt heute Abend wieder."),
        ("es", "Los Sres. García llegan mañana temprano."),
    ],
)
def test_play_inherits_stream_language(language, text, asynchronous):
    stream = TextToAudioStream(_PCMEngine(), tokenizer="rule-based", language=language)
    assert speak(stream, text, asynchronous) == [text]


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_explicit_language_overrides_only_the_current_play(asynchronous):
    text = "Wir besuchen Frl. Schmidt heute Abend wieder."
    stream = TextToAudioStream(_PCMEngine(), tokenizer="rule-based", language="de")
    assert speak(stream, text, asynchronous, language="en") == [
        "Wir besuchen Frl.",
        "Schmidt heute Abend wieder.",
    ]
    assert stream.language == "de"
    assert speak(stream, text, asynchronous) == [text]


def test_default_stream_keeps_english_sentence_splitting():
    text = "We visit Mr. Smith this evening."
    stream = TextToAudioStream(_PCMEngine(), tokenizer="rule-based")
    assert speak(stream, text, False) == [text]
