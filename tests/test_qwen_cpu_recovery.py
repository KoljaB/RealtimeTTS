import threading
import time

import numpy as np
import pytest

from RealtimeTTS import QwenCpuEngine, QwenEngineError, QwenVoice
from RealtimeTTS.engines.qwen_cpu_engine import (
    CPU_ONSET_BASELINE_PROFILE,
    CPU_ONSET_RECOVERY_PROFILE,
)
from tests.test_qwen_engine import FakeVoiceRef, _queued_pcm, _reference_file


class RecoveryBackend:
    def __init__(
        self,
        attempts,
        *,
        producer_alive_after_close=False,
        cancel_on_close=False,
        native_cancel_probe=True,
    ):
        self.library = type("Library", (), {"version": lambda self: "fake-cpu"})()
        self.attempts = list(attempts)
        self.stream_calls = []
        self.validated_onset_profiles = []
        self.closed_generators = 0
        self.producer_alive_after_close = producer_alive_after_close
        self.cancel_on_close = cancel_on_close
        self.native_cancel_probe = native_cancel_probe
        self.last_stream_profile = None
        self.stream_entered = threading.Event()

    def extract_voice_ref(self, audio):
        return FakeVoiceRef()

    def validate_onset_silence_profile(self, profile):
        normalized = str(profile).strip().lower()
        self.validated_onset_profiles.append(normalized)
        return normalized

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        self.stream_entered.set()
        attempt = len(self.stream_calls)
        output = self.attempts[attempt - 1]
        backend = self

        def generate():
            backend.last_stream_profile = {
                "first_callback_perf_counter_ns": time.perf_counter_ns(),
                "first_callback_enter_ms": float(attempt),
                "attempt": attempt,
            }
            try:
                if isinstance(output, BaseException):
                    raise output
                for item in output:
                    if (
                        backend.native_cancel_probe
                        and kwargs["cancel_event"].is_set()
                    ):
                        return
                    yield item, 24000
            finally:
                backend.last_stream_profile = dict(backend.last_stream_profile or {})
                if backend.cancel_on_close:
                    kwargs["cancel_event"].set()
                if backend.producer_alive_after_close is not None:
                    backend.last_stream_profile["producer_alive_after_close"] = (
                        backend.producer_alive_after_close
                    )
                backend.closed_generators += 1

        return generate()

    def close(self):
        pass


def _engine(tmp_path, backend, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return QwenCpuEngine(
        voice=QwenVoice("voice", ref_audio=_reference_file(tmp_path)),
        voice_cache_dir=tmp_path / "cache",
        warmup=False,
        backend_factory=lambda **_kwargs: backend,
        onset_silence_profile=CPU_ONSET_BASELINE_PROFILE,
        onset_silence_recovery=True,
        startup_buffer_ms=0,
        **kwargs,
    )


def _quiet_chunks(count=48):
    return [np.zeros(120, dtype=np.float32) for _ in range(count)]


def _speech_chunk():
    return np.full(240, 0.25, dtype=np.float32)


class NativePauseOnlyCancelEvent:
    """Test event whose is_set method is reserved for native producers."""

    def __init__(self):
        self.is_set_calls = 0

    def cancelled(self):
        return False

    def is_set(self):
        self.is_set_calls += 1
        raise AssertionError("recovery consumer acknowledged a native pause")


def _direct_recovery_stream(engine, cancel_event):
    state = engine._new_onset_recovery_state()
    engine._onset_recovery_state = state
    return state, engine._stream_with_onset_recovery(
        {
            "cancel_event": cancel_event,
            "onset_silence_profile": CPU_ONSET_BASELINE_PROFILE,
        },
        time.perf_counter_ns(),
        state,
    )


def test_quiet_240ms_closes_first_stream_and_retries_once(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "RealtimeTTS.engines.qwen_engine._load_reference_audio",
        lambda _path: np.zeros(2400, dtype=np.float32),
    )
    backend = RecoveryBackend([_quiet_chunks() + [_speech_chunk()], [_speech_chunk()]])
    engine = _engine(tmp_path, backend)
    try:
        assert engine.synthesize("hello") is True
        audio = _queued_pcm(engine)
        assert audio.size > 0
        assert np.max(np.abs(audio)) > 1000
        assert len(backend.stream_calls) == 2
        assert backend.stream_calls[0]["onset_silence_profile"] == CPU_ONSET_BASELINE_PROFILE
        assert backend.stream_calls[1]["onset_silence_profile"] == CPU_ONSET_RECOVERY_PROFILE
        assert backend.validated_onset_profiles == [
            CPU_ONSET_BASELINE_PROFILE,
            CPU_ONSET_RECOVERY_PROFILE,
        ]
        recovery = engine.last_synthesis_profile["onset_recovery"]
        assert recovery["triggered"] is True
        assert recovery["retry_started"] is True
        assert recovery["trigger_ms"] is not None
        assert recovery["abort_ms"] >= recovery["trigger_ms"]
        assert recovery["retry_attempt"]["attempt"] == 2
        assert recovery["first_attempt"]["producer_alive_after_close"] is False
        assert recovery["retry_attempt"]["producer_alive_after_close"] is False
        assert engine.last_synthesis_profile["native"]["attempt"] == 1
        assert engine.last_synthesis_profile["native"]["first_callback_perf_counter_ns"]
        assert backend.closed_generators == 2
    finally:
        engine.shutdown()


def test_speech_in_first_240ms_does_not_retry_and_keeps_pcm(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "RealtimeTTS.engines.qwen_engine._load_reference_audio",
        lambda _path: np.zeros(2400, dtype=np.float32),
    )
    chunks = [_speech_chunk()] + _quiet_chunks(4) + [_speech_chunk()]
    recovery_backend = RecoveryBackend([chunks])
    baseline_backend = RecoveryBackend([chunks])
    recovery_engine = _engine(tmp_path / "recovery", recovery_backend)
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline_engine = QwenCpuEngine(
        voice=QwenVoice("voice", ref_audio=_reference_file(baseline_dir)),
        voice_cache_dir=tmp_path / "baseline-cache",
        warmup=False,
        backend_factory=lambda **_kwargs: baseline_backend,
        onset_silence_profile=CPU_ONSET_BASELINE_PROFILE,
        startup_buffer_ms=0,
    )
    try:
        assert recovery_engine.synthesize("hello") is True
        assert baseline_engine.synthesize("hello") is True
        assert len(recovery_backend.stream_calls) == 1
        assert _queued_pcm(recovery_engine).tobytes() == _queued_pcm(baseline_engine).tobytes()
        recovery = recovery_engine.last_synthesis_profile["onset_recovery"]
        assert recovery["triggered"] is False
        assert recovery["abort_reason"] == "speech_detected"
    finally:
        recovery_engine.shutdown()
        baseline_engine.shutdown()


def test_speech_tail_in_crossing_chunk_is_checked_and_keeps_pcm(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "RealtimeTTS.engines.qwen_engine._load_reference_audio",
        lambda _path: np.zeros(2400, dtype=np.float32),
    )
    crossing_chunk = np.concatenate(
        [np.zeros(5760, dtype=np.float32), _speech_chunk()]
    )
    recovery_backend = RecoveryBackend([[crossing_chunk]])
    baseline_backend = RecoveryBackend([[crossing_chunk]])
    recovery_engine = _engine(tmp_path / "recovery", recovery_backend)
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline_engine = QwenCpuEngine(
        voice=QwenVoice("voice", ref_audio=_reference_file(baseline_dir)),
        voice_cache_dir=tmp_path / "baseline-cache",
        warmup=False,
        backend_factory=lambda **_kwargs: baseline_backend,
        onset_silence_profile=CPU_ONSET_BASELINE_PROFILE,
        startup_buffer_ms=0,
    )
    try:
        assert recovery_engine.synthesize("hello") is True
        assert baseline_engine.synthesize("hello") is True
        assert len(recovery_backend.stream_calls) == 1
        assert _queued_pcm(recovery_engine).tobytes() == _queued_pcm(baseline_engine).tobytes()
        recovery = recovery_engine.last_synthesis_profile["onset_recovery"]
        assert recovery["triggered"] is False
        assert recovery["abort_reason"] == "speech_detected"
    finally:
        recovery_engine.shutdown()
        baseline_engine.shutdown()


def test_icl_and_trim_configurations_never_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "RealtimeTTS.engines.qwen_engine._load_reference_audio",
        lambda _path: np.zeros(2400, dtype=np.float32),
    )
    icl_backend = RecoveryBackend([_quiet_chunks() + [_speech_chunk()]])
    icl_engine = _engine(
        tmp_path / "icl",
        icl_backend,
        clone_mode="auto",
    )
    trim_backend = RecoveryBackend([_quiet_chunks() + [_speech_chunk()]])
    trim_engine = _engine(
        tmp_path / "trim",
        trim_backend,
        trim_silence=False,
    )
    try:
        # Changing the current voice to ICL avoids a second native attempt.
        icl_engine.set_voice(
            QwenVoice(
                "icl",
                ref_audio=_reference_file(tmp_path, "icl-reference.wav"),
                ref_text="reference words",
            )
        )
        assert icl_engine.synthesize("hello") is True
        assert len(icl_backend.stream_calls) == 1
        assert icl_engine.last_synthesis_profile["onset_recovery"]["abort_reason"] == "icl_voice"

        assert trim_engine.synthesize("hello") is True
        assert len(trim_backend.stream_calls) == 1
        assert (
            trim_engine.last_synthesis_profile["onset_recovery"]["abort_reason"]
            == "trim_silence_disabled"
        )
    finally:
        icl_engine.shutdown()
        trim_engine.shutdown()


def test_recovery_requires_known_baseline_and_validates_retry_profile(tmp_path):
    with pytest.raises(ValueError, match="onset_silence_recovery requires"):
        QwenCpuEngine(
            onset_silence_recovery=True,
            backend_factory=lambda **_kwargs: pytest.fail("backend must not load"),
        )

    class NoProfileBackend(RecoveryBackend):
        def validate_onset_silence_profile(self, profile):
            if str(profile).strip().lower() == CPU_ONSET_RECOVERY_PROFILE:
                raise RuntimeError("CPU recovery profile is unavailable")
            return super().validate_onset_silence_profile(profile)

    with pytest.raises(QwenEngineError, match="CPU recovery profile is unavailable"):
        QwenCpuEngine(
            onset_silence_recovery=True,
            onset_silence_profile=CPU_ONSET_BASELINE_PROFILE,
            backend_factory=lambda **_kwargs: NoProfileBackend([]),
            warmup=False,
        )


def test_cancellation_before_quiet_window_never_starts_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "RealtimeTTS.engines.qwen_engine._load_reference_audio",
        lambda _path: np.zeros(2400, dtype=np.float32),
    )
    release = threading.Event()

    def endless_quiet():
        while not release.is_set():
            time.sleep(0.01)
            yield np.zeros(120, dtype=np.float32)

    backend = RecoveryBackend([endless_quiet()])
    engine = _engine(tmp_path, backend)
    result = []
    worker = threading.Thread(target=lambda: result.append(engine.synthesize("hello")))
    worker.start()
    try:
        assert backend.stream_entered.wait(1)
        time.sleep(0.02)
        engine.stop()
        release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert result == [True]
        assert len(backend.stream_calls) == 1
        recovery = engine.last_synthesis_profile["onset_recovery"]
        assert recovery["retry_started"] is False
        assert recovery["abort_reason"] == "cancelled"
    finally:
        release.set()
        engine.stop()
        worker.join(2)
        engine.shutdown()


def test_cancellation_during_first_close_never_starts_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "RealtimeTTS.engines.qwen_engine._load_reference_audio",
        lambda _path: np.zeros(2400, dtype=np.float32),
    )
    backend = RecoveryBackend(
        [_quiet_chunks(), [_speech_chunk()]], cancel_on_close=True
    )
    engine = _engine(tmp_path, backend)
    try:
        assert engine.synthesize("hello") is True
        assert len(backend.stream_calls) == 1
        recovery = engine.last_synthesis_profile["onset_recovery"]
        assert recovery["triggered"] is True
        assert recovery["retry_attempted"] is False
        assert recovery["abort_reason"] == "cancelled_before_retry"
    finally:
        engine.shutdown()


def test_recovery_consumer_never_acks_native_pause_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "RealtimeTTS.engines.qwen_engine._load_reference_audio",
        lambda _path: np.zeros(2400, dtype=np.float32),
    )
    backend = RecoveryBackend(
        [_quiet_chunks() + [_speech_chunk()], [_speech_chunk()]],
        native_cancel_probe=False,
    )
    engine = _engine(tmp_path, backend)
    cancel_event = NativePauseOnlyCancelEvent()
    try:
        state, stream = _direct_recovery_stream(engine, cancel_event)
        assert list(stream)
        assert cancel_event.is_set_calls == 0
        assert state["retry_started"] is True
        assert len(backend.stream_calls) == 2
    finally:
        engine.shutdown()


def test_recovery_does_not_close_or_retry_while_pause_is_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "RealtimeTTS.engines.qwen_engine._load_reference_audio",
        lambda _path: np.zeros(2400, dtype=np.float32),
    )
    backend = RecoveryBackend(
        [_quiet_chunks() + [_speech_chunk()]],
        native_cancel_probe=False,
    )
    engine = _engine(tmp_path, backend)
    cancel_event = NativePauseOnlyCancelEvent()
    try:
        engine._pause_requested = True
        state, stream = _direct_recovery_stream(engine, cancel_event)
        assert list(stream)
        assert len(backend.stream_calls) == 1
        assert backend.closed_generators == 1
        assert state["triggered"] is False
        assert state["retry_started"] is False
        assert state["abort_reason"] == "paused_before_retry"
        assert cancel_event.is_set_calls == 0
    finally:
        engine.shutdown()


@pytest.mark.parametrize("producer_state", [None, True])
def test_recovery_refuses_unverified_native_join(tmp_path, monkeypatch, producer_state):
    monkeypatch.setattr(
        "RealtimeTTS.engines.qwen_engine._load_reference_audio",
        lambda _path: np.zeros(2400, dtype=np.float32),
    )
    backend = RecoveryBackend(
        [_quiet_chunks(), [_speech_chunk()]],
        producer_alive_after_close=producer_state,
    )
    engine = _engine(tmp_path, backend)
    try:
        assert engine.synthesize("hello") is False
        message = str(engine.last_error)
        if producer_state is None:
            assert "producer_alive_after_close" in message
        else:
            assert "producer remained alive" in message
        assert len(backend.stream_calls) == 1
        assert engine.last_synthesis_profile["onset_recovery"]["retry_attempted"] is False
    finally:
        engine.shutdown()


def test_failed_retry_remains_visible_and_is_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "RealtimeTTS.engines.qwen_engine._load_reference_audio",
        lambda _path: np.zeros(2400, dtype=np.float32),
    )
    backend = RecoveryBackend(
        [_quiet_chunks(), RuntimeError("retry native failure")]
    )
    engine = _engine(tmp_path, backend)
    try:
        assert engine.synthesize("hello") is False
        assert "retry native failure" in str(engine.last_error)
        assert len(backend.stream_calls) == 2
        recovery = engine.last_synthesis_profile["onset_recovery"]
        assert recovery["retry_attempted"] is True
        assert recovery["retry_started"] is True
        assert recovery["retry_attempt"]["attempt"] == 2
        assert engine.last_synthesis_profile["native"]["attempt"] == 1
        assert backend.closed_generators == 2
    finally:
        engine.shutdown()
