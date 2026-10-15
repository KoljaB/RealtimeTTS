"""play() splits sentences with the language the stream was created with."""

import pytest

from RealtimeTTS import TextToAudioStream
from tests.test_text_to_stream_alignment import _FakePyAudio, _PCMEngine
import RealtimeTTS.stream_player as stream_player


@pytest.mark.parametrize(
    ("language", "text"),
    [
        # "Frl." and "Sres." are abbreviations only in the German and Spanish rules
        ("de", "Wir besuchen Frl. Schmidt heute Abend wieder."),
        ("es", "Los Sres. García llegan mañana temprano."),
    ],
)
def test_play_uses_the_stream_language(monkeypatch, language, text):
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)
    fragments = []
    stream = TextToAudioStream(_PCMEngine(), tokenizer="rule-based", language=language)
    stream.feed(text).play(
        before_sentence_synthesized=fragments.append,
        minimum_sentence_length=1,
        minimum_first_fragment_length=1,
    )
    assert fragments == [text]
