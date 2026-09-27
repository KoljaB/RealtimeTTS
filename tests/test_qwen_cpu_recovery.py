"""Quiet native audio is trimmed without cancelling or restarting synthesis."""
import numpy as np

from RealtimeTTS import QwenCpuEngine, QwenVoice
from tests.test_qwen_engine import SequenceBackend, _queued_pcm, _reference_file


def test_long_quiet_onset_continues_original_stream_and_only_queues_speech(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "RealtimeTTS.engines.qwen_engine._load_reference_audio",
        lambda _: np.zeros(2400, dtype=np.float32),
    )
    quiet = [np.zeros(1920, dtype=np.float32) for _ in range(8)]
    speech = [np.full(1920, 0.25, dtype=np.float32) for _ in range(8)]
    backend = SequenceBackend(quiet + speech)
    profile = "qwen3_tts_12hz_0_6b_base_q8_v1"
    engine = QwenCpuEngine(
        voice=QwenVoice("voice", ref_audio=_reference_file(tmp_path)),
        voice_cache_dir=tmp_path / "cache",
        warmup=False,
        backend_factory=lambda **kwargs: backend,
        onset_silence_profile=profile,
        startup_buffer_ms=80,
    )
    try:
        assert engine.synthesize("Continue the original speech.")
        audio = _queued_pcm(engine)
        assert len(backend.stream_calls) == 1
        assert backend.stream_calls[0]["onset_silence_profile"] == profile
        assert backend.validated_onset_profiles == [profile]
        assert engine.last_synthesis_profile["leading_trimmed_ms"] == 625
        assert engine.last_synthesis_profile["startup_speech_detected"] is True
        # Only the explicit 15 ms pre-roll survives, followed by all the speech.
        assert audio.size == 8 * 1920 + 360
        assert np.max(np.abs(audio[:2400])) > 7000
    finally:
        engine.shutdown()
