"""Opt-in continuation guard stops one native stream and preserves a faded tail."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from RealtimeTTS import QwenVoice
from RealtimeTTS.engines.qwen_engine import _float_to_pcm16
from tests.test_qwen_engine import SequenceBackend, _engine, _queued_pcm, _reference_file

RATE = 24000


def tone(seconds, hz, amplitude):
    t = np.arange(round(seconds * RATE), dtype=np.float32) / RATE
    return amplitude * np.sin(2 * np.pi * hz * t)


def make_engine(tmp_path, monkeypatch, audio, *, guard=True, block=480):
    monkeypatch.setattr("RealtimeTTS.engines.qwen_engine._load_reference_audio",
                        lambda _: np.zeros(2400, dtype=np.float32))
    backend = SequenceBackend([audio[i:i + block] for i in range(0, len(audio), block)])
    engine = _engine(tmp_path, backend,
                     voice=QwenVoice("voice", ref_audio=_reference_file(tmp_path)),
                     trim_silence=False, startup_buffer_ms=0, acoustic_guard=guard)
    return engine, backend


def test_noise_tail_stops_native_once_and_fades_without_touching_start(tmp_path, monkeypatch):
    audio = np.concatenate((tone(1.2, 1200, .2), np.zeros(RATE * 2),
                            tone(2, 120, .01))).astype(np.float32)
    engine, backend = make_engine(tmp_path, monkeypatch, audio)
    try:
        assert engine.synthesize("A sufficiently long clean phrase.")
        output = _queued_pcm(engine)
        profile = engine.last_synthesis_profile
        assert len(backend.stream_calls) == 1  # No automatic retry.
        assert profile["finish_reason"] == "acoustic_noise_tail"
        assert not profile["cancelled"]  # Graceful finish retains the PCM fade.
        assert profile["ending_fade_samples"] == 600
        assert output[-1] == 0
        assert len(output) < len(audio)
        assert output[:RATE].tobytes() == _float_to_pcm16(audio[:RATE])
        decision = profile["acoustic_guard"]["decision"]
        assert decision["accepted_speech_s"] >= 1
        assert decision["evidence_s"] - decision["onset_s"] <= .52
    finally:
        engine.shutdown()


@pytest.mark.parametrize("guard", [False, True])
def test_short_speech_then_noise_is_not_stopped_or_retried(tmp_path, monkeypatch, guard):
    audio = np.concatenate((tone(.6, 1200, .2), np.zeros(RATE * 2),
                            tone(2, 120, .01))).astype(np.float32)
    engine, backend = make_engine(tmp_path, monkeypatch, audio, guard=guard)
    try:
        assert engine.synthesize("Short phrase")
        output = _queued_pcm(engine)
        assert len(backend.stream_calls) == 1
        assert engine.last_synthesis_profile["finish_reason"] == "completed"
        assert output.tobytes() == _float_to_pcm16(audio)
    finally:
        engine.shutdown()


def test_gate_does_not_carry_between_fragments(tmp_path, monkeypatch):
    audio = tone(1.2, 1200, .2).astype(np.float32)
    engine, backend = make_engine(tmp_path, monkeypatch, audio)
    try:
        assert engine.synthesize("First fragment")
        assert engine.last_synthesis_profile["acoustic_guard"]["armed"]
        _queued_pcm(engine)
        second = np.concatenate((tone(.4, 1200, .2), np.zeros(RATE * 2),
                                 tone(2, 120, .01))).astype(np.float32)
        backend.chunks = [second[i:i + 480] for i in range(0, len(second), 480)]
        assert engine.synthesize("Second short fragment")
        assert engine.last_synthesis_profile["finish_reason"] == "completed"
        assert _queued_pcm(engine).tobytes() == _float_to_pcm16(second)
    finally:
        engine.shutdown()


def test_clean_long_fragment_is_bit_identical_without_normal_trimming(tmp_path, monkeypatch):
    audio = tone(2, 1200, .2).astype(np.float32)
    engine, _ = make_engine(tmp_path, monkeypatch, audio)
    try:
        assert engine.synthesize("Clean speech without an anomaly.")
        assert engine.last_synthesis_profile["acoustic_guard"]["armed"]
        assert engine.last_synthesis_profile["ending_fade_samples"] == 0
        assert _queued_pcm(engine).tobytes() == _float_to_pcm16(audio)
    finally:
        engine.shutdown()


def test_decision_inside_large_unaligned_chunk_retains_prefix_and_cancels_alignment(tmp_path, monkeypatch):
    import threading
    audio = np.concatenate((tone(1.2, 1200, .2), np.zeros(RATE * 2),
                            tone(2, 120, .01))).astype(np.float32)
    engine, backend = make_engine(tmp_path, monkeypatch, audio, block=47999)
    class Session:
        stop_requested = threading.Event()
        finish_reason = None
        samples = 0
        cancelled = None
        def feed(self, block):
            self.samples += len(block)
        def finish(self, *, cancelled=False):
            self.cancelled = cancelled
    session = Session()
    engine.streaming_aligner = SimpleNamespace(start=lambda *_args, **_kwargs: session)
    try:
        assert engine.synthesize("This fragment must not be declared fully aligned.")
        output = _queued_pcm(engine)
        assert len(backend.stream_calls) == 1
        assert len(output) == round(3.2 * RATE)  # Noise starts within the second native chunk.
        assert output[:RATE].tobytes() == _float_to_pcm16(audio[:RATE])
        assert session.samples == len(output)
        assert session.cancelled is True
    finally:
        engine.streaming_aligner = None
        engine.shutdown()


def test_server_guard_is_explicit_and_forwarded_to_cpu(monkeypatch):
    import RealtimeTTS.qwen_server as server
    assert not server.build_argument_parser().parse_args([]).acoustic_guard
    received = {}

    class Reached(Exception):
        pass

    def cpu_engine(**kwargs):
        received.update(kwargs)
        raise Reached

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace())
    monkeypatch.setattr(server, "QwenCpuEngine", cpu_engine)
    with pytest.raises(Reached):
        server.main(["--device", "cpu", "--acoustic-guard"])
    assert received["acoustic_guard"] is True
