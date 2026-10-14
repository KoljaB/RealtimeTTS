import builtins
import math

import numpy as np
import pytest

from RealtimeTTS import _audio_formats as formats
from RealtimeTTS.audio_tempo import SpeedChange, TempoProcessor, validate_speed


def _render(processor, payload, sizes):
    chunks = []
    cursor = 0
    index = 0
    while cursor < len(payload):
        size = sizes[index % len(sizes)]
        chunks.extend(processor.feed(payload[cursor:cursor + size]))
        cursor += size
        index += 1
    chunks.extend(processor.finish())
    return b"".join(chunks)


def _sine(rate=24_000, seconds=0.8, frequency=440.0):
    count = int(rate * seconds)
    t = np.arange(count, dtype=np.float32) / rate
    return np.sin(2.0 * np.pi * frequency * t).astype("<f4")


def test_speed_validation_and_immutable_marker():
    for value in (0.25, 1.0, 4.0):
        assert validate_speed(value) == value
        assert SpeedChange(value).speed == value
    for value in (0.2499, 4.0001, math.nan, math.inf, -math.inf, True):
        with pytest.raises(ValueError):
            validate_speed(value)
    marker = SpeedChange(1.25)
    with pytest.raises((AttributeError, TypeError)):
        marker.speed = 1.5


def test_speed_one_is_a_byte_bypass_without_importing_av(monkeypatch):
    real_import = builtins.__import__

    def reject_av(name, *args, **kwargs):
        if name == "av" or name.startswith("av."):
            raise AssertionError("speed-one bypass imported PyAV")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_av)
    payload = b"\x01\x02\x03\x04\x05\x06"
    processor = TempoProcessor(formats.INT16, 1, 24_000, 1.0)
    assert b"".join(processor.feed(payload[:1])) == b""
    assert b"".join(processor.feed(payload[1:])) == payload
    assert b"".join(processor.finish()) == b""


@pytest.mark.parametrize("stream_format,dtype", [
    (formats.INT16, np.dtype("<i2")),
    (formats.FLOAT32, np.dtype("<f4")),
])
def test_tempo_preserves_duration_format_and_pitch(stream_format, dtype):
    pytest.importorskip("av")
    source = _sine()
    if stream_format == formats.INT16:
        payload = np.rint(source * 32767.0).astype(dtype).tobytes()
    else:
        payload = source.astype(dtype).tobytes()

    output = _render(
        TempoProcessor(stream_format, 1, 24_000, 1.25),
        payload,
        [37, 991, 5, 4096],
    )
    values = np.frombuffer(output, dtype=dtype)
    assert abs(len(values) / 24_000.0 - 0.8 / 1.25) < 0.035

    middle = values[len(values) // 4: 3 * len(values) // 4].astype(np.float32)
    if stream_format == formats.INT16:
        middle /= 32767.0
    spectrum = np.fft.rfft(middle * np.hanning(len(middle)))
    peak_hz = (np.argmax(np.abs(spectrum[1:])) + 1) * 24_000.0 / len(middle)
    assert abs(peak_hz - 440.0) < 18.0


def test_arbitrary_chunk_splits_and_short_finish_tail_are_stable():
    pytest.importorskip("av")
    source = _sine(seconds=0.18)
    whole = source.astype("<f4").tobytes()
    rendered_whole = _render(
        TempoProcessor(formats.FLOAT32, 1, 24_000, 0.8),
        whole,
        [len(whole)],
    )
    rendered_split = _render(
        TempoProcessor(formats.FLOAT32, 1, 24_000, 0.8),
        whole,
        [1, 7, 31, 2, 409, 13],
    )
    whole_values = np.frombuffer(rendered_whole, dtype="<f4")
    split_values = np.frombuffer(rendered_split, dtype="<f4")
    assert abs(len(whole_values) - len(split_values)) <= 2
    assert abs(len(split_values) / 24_000.0 - 0.18 / 0.8) < 0.035
    np.testing.assert_allclose(split_values, whole_values, rtol=0, atol=2e-3)

@pytest.mark.parametrize("rate", (16_000, 24_000, 48_000))
@pytest.mark.parametrize("speed", (0.25, 1.5, 4.0))
def test_short_segments_keep_signal_at_supported_rates(rate, speed):
    count = round(rate * 0.012)
    t = np.arange(count, dtype=np.float32) / rate
    source = (0.7 * np.sin(2.0 * np.pi * 440.0 * t)).astype("<f4")
    output = _render(
        TempoProcessor(formats.FLOAT32, 1, rate, speed),
        source.tobytes(),
        [5, 11, 19],
    )
    values = np.frombuffer(output, dtype="<f4")
    assert len(values) > 0
    assert np.max(np.abs(values)) > 0.1
    assert np.sqrt(np.mean(values * values)) > 0.03


def test_finish_retains_signal_at_segment_tail():
    rate = 24_000
    count = round(rate * 0.12)
    t = np.arange(count, dtype=np.float32) / rate
    source = np.zeros(count, dtype="<f4")
    source[-round(rate * 0.04):] = (
        0.7 * np.sin(2.0 * np.pi * 880.0 * t[-round(rate * 0.04):])
    ).astype("<f4")
    output = _render(
        TempoProcessor(formats.FLOAT32, 1, rate, 1.25),
        source.tobytes(),
        [113, 701, 17],
    )
    values = np.frombuffer(output, dtype="<f4")
    tail = values[-min(len(values), 256):]
    assert np.max(np.abs(tail)) > 0.2
    assert np.sqrt(np.mean(tail * tail)) > 0.05


def test_stereo_pcm16_and_cancel_discards_old_tail():
    pytest.importorskip("av")
    rate = 24_000
    mono = (_sine(rate=rate, seconds=0.22) * 0.5).astype("<f4")
    stereo = np.column_stack((mono, -mono)).astype("<f4").reshape(-1)
    payload = np.rint(stereo * 32767.0).astype("<i2").tobytes()

    processor = TempoProcessor(formats.INT16, 2, rate, 1.2)
    list(processor.feed(payload[:1000]))
    processor.cancel()
    assert processor._graph is None
    with pytest.raises(RuntimeError, match="already been finished"):
        list(processor.feed(payload))

    fresh = _render(
        TempoProcessor(formats.INT16, 2, rate, 1.2),
        payload,
        [17, 503, 2_001],
    )
    assert len(fresh) > 0


def test_partial_frame_and_feed_after_finish_are_clear_errors():
    pytest.importorskip("av")
    processor = TempoProcessor(formats.FLOAT32, 2, 24_000, 1.1)
    list(processor.feed(b"\x00"))
    with pytest.raises(ValueError, match="trailing bytes"):
        list(processor.finish())

    processor = TempoProcessor(formats.FLOAT32, 1, 24_000, 1.1)
    list(processor.finish())
    with pytest.raises(RuntimeError, match="already been finished"):
        list(processor.feed(b"\x00\x00\x00\x00"))
