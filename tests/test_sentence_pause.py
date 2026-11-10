"""A terminator inside closing quotes or brackets still ends the sentence."""

import pytest

import RealtimeTTS.stream_player as stream_player
from RealtimeTTS import TextToAudioStream
from RealtimeTTS.qwen_server import QwenHttpServer
from tests.test_text_to_stream_alignment import _FakePyAudio, _PCMEngine

RATE = 16000
SENTENCE_S = 0.5
COMMA_S = 0.1


def _pcm_bytes(seconds: float) -> int:
    return int(RATE * seconds) * 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('She said "hello."', 0.30),
        ("She said “hello.”", 0.30),
        ("(Yes.)", 0.30),
        ("Hello.", 0.30),
        ("Hello,", 0.15),
        ('She said "hello,"', 0.0),
        ("Wait)", 0.15),
    ],
)
def test_qwen_server_preserves_punctuation_pause_class(text, expected):
    server = object.__new__(QwenHttpServer)
    server.sentence_silence_duration = 0.30
    server.comma_silence_duration = 0.15

    assert server.fragment_silence_duration(text) == expected


def _played_bytes(monkeypatch, sentence_s: float, comma_s: float) -> int:
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)
    received: list[bytes] = []
    stream = TextToAudioStream(_PCMEngine(), tokenizer="rule-based")
    stream.feed('She said "hello."').play(
        on_audio_chunk=received.append,
        sentence_silence_duration=sentence_s,
        comma_silence_duration=comma_s,
        default_silence_duration=0.0,
        minimum_sentence_length=1,
        minimum_first_fragment_length=1,
        muted=True,
    )
    return sum(len(chunk) for chunk in received)


def test_play_uses_sentence_pause_and_keeps_zero_durations(monkeypatch):
    assert _played_bytes(monkeypatch, SENTENCE_S, COMMA_S) == (
        _pcm_bytes(1.0) + _pcm_bytes(SENTENCE_S)
    )
    assert _played_bytes(monkeypatch, 0.0, 0.0) == _pcm_bytes(1.0)
