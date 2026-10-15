"""Text fed while play() runs is played with the options play() was called with."""

from RealtimeTTS import TextToAudioStream
from tests.test_text_to_stream_alignment import _FakePyAudio, _PCMEngine
import RealtimeTTS.stream_player as stream_player


def test_text_fed_during_play_keeps_the_play_options(monkeypatch):
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)
    stream = TextToAudioStream(_PCMEngine(), tokenizer="rule-based")
    fragments = []
    tokenized = []

    def before_sentence_synthesized(sentence):
        fragments.append(sentence)
        if len(fragments) == 1:
            # The first sentence has been read, so these go to the recursive play() call
            stream.feed("Second reply arrives. ")
            stream.feed("Third part follows.")

    def tokenize_sentences(text):
        tokenized.append(text)
        return [text]

    stream.feed("First reply here.").play(
        before_sentence_synthesized=before_sentence_synthesized,
        tokenize_sentences=tokenize_sentences,
        minimum_sentence_length=1,
        minimum_first_fragment_length=1,
    )

    assert fragments == ["First reply here.", "Second reply arrives.", "Third part follows."]
    assert any(text.startswith("Third part") for text in tokenized)
