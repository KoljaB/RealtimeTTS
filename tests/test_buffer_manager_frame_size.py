"""Unit tests for AudioBufferManager sample accounting.

The buffered-sample counter must use the same bytes-per-frame divisor when
audio is added and when it is removed, so that feeding N seconds of audio and
then draining it returns the counter to zero for every supported format and
channel count -- not only the default 16-bit mono case.
"""

import queue

import pyaudio
import pytest

from RealtimeTTS.stream_player import AudioBufferManager, AudioConfiguration

RATE = 24000


@pytest.mark.parametrize(
    "fmt,channels,bytes_per_sample",
    [
        (pyaudio.paInt16, 1, 2),  # default / control
        (pyaudio.paFloat32, 1, 4),  # XTTS/Coqui, Parler, Cartesia
        (pyaudio.paInt16, 2, 2),  # stereo
        (pyaudio.paFloat32, 2, 4),
    ],
)
def test_buffered_seconds_round_trip(fmt, channels, bytes_per_sample):
    config = AudioConfiguration(format=fmt, channels=channels, rate=RATE)
    manager = AudioBufferManager(queue.Queue(), queue.Queue(), config)

    one_second = b"\x00" * (RATE * channels * bytes_per_sample)
    manager.add_to_buffer(one_second)
    assert manager.get_buffered_seconds(RATE) == pytest.approx(1.0)

    ok, _ = manager.get_from_buffer()
    assert ok
    assert manager.get_buffered_seconds(RATE) == pytest.approx(0.0)
