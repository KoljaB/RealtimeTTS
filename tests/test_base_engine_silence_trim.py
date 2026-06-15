import numpy as np

from RealtimeTTS.engines.base_engine import BaseEngine


def test_trim_silence_start_ignores_single_sample_spike():
    engine = BaseEngine()
    sample_rate = 24000
    threshold = 80 / 32767
    audio = np.zeros(sample_rate // 5, dtype=np.float32)
    audio[0] = threshold * 3
    speech_start = int(sample_rate * 0.16)
    audio[speech_start:speech_start + 480] = threshold * 4

    trimmed = engine.trim_silence_start(
        audio,
        sample_rate=sample_rate,
        silence_threshold=threshold,
        extra_ms=0,
        fade_in_ms=0,
    )

    assert len(audio) - len(trimmed) == speech_start
    assert trimmed[0] == audio[speech_start]


def test_trim_silence_start_keeps_immediate_audio():
    engine = BaseEngine()
    sample_rate = 24000
    threshold = 80 / 32767
    audio = np.full(480, threshold * 4, dtype=np.float32)

    trimmed = engine.trim_silence_start(
        audio,
        sample_rate=sample_rate,
        silence_threshold=threshold,
        extra_ms=0,
        fade_in_ms=0,
    )

    assert np.array_equal(trimmed, audio)


def test_trim_silence_only_trims_stream_start_once():
    engine = BaseEngine()
    sample_rate = 24000
    threshold = 80 / 32767
    engine.synthesize("hello")

    silent_chunk = np.zeros(1920, dtype=np.float32)
    silent_chunk[0] = threshold * 3
    first = engine._trim_silence(
        silent_chunk,
        sample_rate=sample_rate,
        silence_threshold=threshold,
        extra_start_ms=0,
        extra_end_ms=0,
        fade_in_ms=0,
        fade_out_ms=0,
    )
    assert first.size == 0

    speech = np.full(480, threshold * 4, dtype=np.float32)
    second_chunk = np.concatenate([silent_chunk, speech])
    second = engine._trim_silence(
        second_chunk,
        sample_rate=sample_rate,
        silence_threshold=threshold,
        extra_start_ms=0,
        extra_end_ms=0,
        fade_in_ms=0,
        fade_out_ms=0,
    )
    assert np.array_equal(second, speech)

    third_chunk = np.concatenate([silent_chunk, speech])
    third = engine._trim_silence(
        third_chunk,
        sample_rate=sample_rate,
        silence_threshold=threshold,
        extra_start_ms=0,
        extra_end_ms=0,
        fade_in_ms=0,
        fade_out_ms=0,
    )
    assert np.array_equal(third, third_chunk)


def test_trim_silence_can_leave_trailing_audio_untouched():
    engine = BaseEngine()
    sample_rate = 24000
    threshold = 80 / 32767
    engine.synthesize("hello")
    leading_silence = np.zeros(1920, dtype=np.float32)
    speech = np.full(480, threshold * 4, dtype=np.float32)
    trailing_silence = np.zeros(1920, dtype=np.float32)
    audio = np.concatenate([leading_silence, speech, trailing_silence])

    trimmed = engine._trim_silence(
        audio,
        sample_rate=sample_rate,
        silence_threshold=threshold,
        extra_start_ms=0,
        extra_end_ms=0,
        fade_in_ms=0,
        fade_out_ms=0,
    )

    assert np.array_equal(trimmed, np.concatenate([speech, trailing_silence]))


def test_zero_duration_fades_are_noops():
    engine = BaseEngine()
    audio = np.array([1.0, 1.0], dtype=np.float32)

    assert np.array_equal(engine.apply_fade_in(audio, 24000, 0), audio)
    assert np.array_equal(engine.apply_fade_out(audio, 24000, 0), audio)
