import queue
import threading
import types

import numpy as np

from RealtimeTTS.stream_player import StreamPlayer


class _EmptyBuffer:
    def empty(self):
        return True


class _StoppingBufferManager:
    def __init__(self, player):
        self.player = player
        self.audio_buffer = _EmptyBuffer()
        self.cleared = False

    def get_from_buffer(self):
        self.player.playback_active = False
        return False, None

    def clear_buffer(self):
        self.cleared = True


class _FakeAudioStream:
    def __init__(self):
        self.closed = False

    def close_stream(self):
        self.closed = True


def test_process_buffer_marks_playback_inactive_on_exit():
    player = object.__new__(StreamPlayer)
    player.playback_active = True
    player.immediate_stop = threading.Event()
    player.on_playback_stop = None
    player.buffer_manager = _StoppingBufferManager(player)

    player._process_buffer()

    assert player.playback_active is False


def test_immediate_stop_joins_and_cleans_up_playback_thread():
    player = object.__new__(StreamPlayer)
    player.playback_active = True
    player.immediate_stop = threading.Event()
    player.audio_stream = _FakeAudioStream()
    player.buffer_manager = _StoppingBufferManager(player)

    def playback_worker():
        player.immediate_stop.wait(timeout=1.0)
        player.playback_active = False

    player.playback_thread = threading.Thread(target=playback_worker)
    player.playback_thread.start()

    player.stop(immediate=True)

    assert player.playback_active is False
    assert player.audio_stream.closed is True
    assert player.buffer_manager.cleared is True
    assert player.playback_thread is None
    assert player.immediate_stop.is_set() is False


class _CountingStream:
    """Records the total number of bytes handed to the audio device."""

    def __init__(self):
        self.bytes_written = 0
        self.data = b""

    def write(self, data):
        self.bytes_written += len(data)
        self.data += data


def _play_one_second(source_rate, device_rate, chunk=None, channels=1):
    """Feed one second of int16 audio at ``source_rate`` (mono silence unless
    ``chunk`` is given) into a player whose device runs at ``device_rate`` and
    return (player, stream)."""
    import pyaudio

    player = object.__new__(StreamPlayer)
    player.muted = False
    player.volume = 1.0
    player.first_chunk_played = True
    player.on_playback_start = None
    player.on_word_spoken = None
    player.on_audio_chunk = None
    player.seconds_played = 0.0
    player.timings = queue.Queue()
    player.timings_list = []
    player.pause_event = threading.Event()
    player.immediate_stop = threading.Event()

    config = types.SimpleNamespace(
        format=pyaudio.paInt16,
        rate=source_rate,
        channels=channels,
        playout_chunk_size=8192,
    )
    stream = _CountingStream()
    player.audio_stream = types.SimpleNamespace(
        config=config,
        actual_sample_rate=device_rate,
        pyaudio_instance=types.SimpleNamespace(get_sample_size=lambda fmt: 2),
        stream=stream,
    )

    if chunk is None:
        chunk = np.zeros(source_rate, dtype=np.int16).tobytes()  # 1.0 s mono
    player._play_wav_chunk(chunk)
    return player, stream


def test_seconds_played_tracks_device_output_rate_when_resampling():
    # Device opens at 48 kHz while the engine emits 24 kHz, so each chunk is
    # resampled up before being written. seconds_played must count the bytes
    # that actually reached the device (at the device rate), not divide the
    # resampled byte count by the source rate -- doing the latter scales the
    # playback clock (and the word-timing callbacks it drives) by the resample
    # ratio, here 2x.
    sample_width, channels = 2, 1
    player, stream = _play_one_second(source_rate=24000, device_rate=48000)
    expected = stream.bytes_written / (48000 * sample_width * channels)
    assert expected == 1.0  # one second really was written to the device
    assert abs(player.seconds_played - expected) < 1e-9


def test_seconds_played_unchanged_without_resampling():
    sample_width, channels = 2, 1
    player, stream = _play_one_second(source_rate=24000, device_rate=24000)
    expected = stream.bytes_written / (24000 * sample_width * channels)
    assert abs(player.seconds_played - expected) < 1e-9


def test_resampling_keeps_stereo_channels_apart():
    # Stereo engines (MOSS-TTS defaults to 2 channels) hand interleaved L/R
    # frames to the player. Resampling them as one mono signal blends the two
    # channels: a tone on the left leaks into a silent right channel.
    t = np.arange(48000) / 48000
    left = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    stereo = np.stack([left, np.zeros_like(left)], axis=1)
    _, stream = _play_one_second(
        source_rate=48000, device_rate=24000, chunk=stereo.tobytes(), channels=2
    )
    out = np.frombuffer(stream.data, dtype=np.int16).reshape(-1, 2)
    assert abs(len(out) - 24000) <= 1
    assert np.abs(out[:, 1]).max() <= 1  # right channel stays silent
    assert np.abs(out[:, 0]).max() > 15000  # left keeps its level
