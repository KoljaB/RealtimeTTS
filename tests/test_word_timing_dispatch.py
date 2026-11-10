"""StreamPlayer must dispatch every already-elapsed word timing per audio write.

Word timings for a whole sentence are enqueued at once (forced alignment in
text_to_stream.py enqueues all words of a sentence, and several engines push
multiple TimingInfo entries). While a single audio chunk is written,
``seconds_played`` can advance past the start_time of several queued words.
The ``on_word_spoken`` callback must fire for all of them, in order, not just
for one word per write.
"""

import queue

import pyaudio

import RealtimeTTS.stream_player as stream_player
from RealtimeTTS.stream_player import AudioConfiguration, StreamPlayer
from RealtimeTTS.engines.base_engine import TimingInfo

from tests.test_text_to_stream_alignment import _FakePyAudio


def _make_player(monkeypatch):
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)
    config = AudioConfiguration(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        # One single sub-chunk covers the whole written chunk below.
        playout_chunk_size=1_000_000,
    )
    spoken = []
    player = StreamPlayer(
        audio_buffer=queue.Queue(),
        timings=queue.Queue(),
        config=config,
        on_word_spoken=spoken.append,
    )

    class _Stream:
        def write(self, chunk):
            pass

    player.audio_stream.stream = _Stream()
    player.audio_stream.actual_sample_rate = 16000
    return player, spoken


def test_all_elapsed_words_dispatched_in_single_write(monkeypatch):
    player, spoken = _make_player(monkeypatch)

    # Four words, each already started well before the audio we are about to
    # write finishes playing.
    for i in range(4):
        player.timings.put(TimingInfo(0.01 * i, 0.01 * i + 0.005, f"w{i}"))

    # Write one second of 16 kHz mono int16 audio (32000 bytes). After this
    # write seconds_played == 1.0, so all four words (start_time <= 0.03) have
    # elapsed and must have been spoken.
    one_second = b"\x00\x00" * 16000
    player._play_wav_chunk(one_second)

    assert [t.word for t in spoken] == ["w0", "w1", "w2", "w3"]
