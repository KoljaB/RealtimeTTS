import os
import sys
from types import SimpleNamespace

import pytest
import numpy as np

from RealtimeTTS import QwenCpuEngine, QwenEngineError, QwenVoice
from tests.test_qwen_engine import FakeBackend, _queued_pcm, _reference_file


@pytest.mark.parametrize("threads", [0, -1, 257, True, 2.5, "8"])
def test_invalid_thread_count_fails_before_loading(threads):
    with pytest.raises(ValueError, match="cpu_threads"):
        QwenCpuEngine(cpu_threads=threads)


def test_cpu_engine_rejects_gpu_binding_before_constructing_context(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("GPU context must not be created")

    monkeypatch.setitem(sys.modules, "qwentts_cpp", SimpleNamespace(QwenTTS=forbidden))
    with pytest.raises(QwenEngineError, match="CPU wheel"):
        QwenCpuEngine(warmup=False)


def test_cpu_engine_streams_both_clone_modes_without_changing_environment(tmp_path, monkeypatch):
    # This test selects six workers deliberately, independent of CI host size.
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setenv("GGML_BACKEND", "CUDA")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    before = dict(os.environ)
    backend = FakeBackend()
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return backend

    engine = QwenCpuEngine(
        cpu_threads=6, backend_factory=factory, warmup=False,
        voice_cache_dir=tmp_path / "cache", startup_buffer_ms=40, clone_mode="auto",
    )
    try:
        monkeypatch.setattr(
            "RealtimeTTS.engines.qwen_engine._load_reference_audio",
            lambda _: np.zeros(2400, dtype=np.float32),
        )
        wav = _reference_file(tmp_path)
        engine.set_voice(QwenVoice("speaker", ref_audio=wav))
        assert engine.synthesize("The CPU voice is ready.")
        assert _queued_pcm(engine).size
        assert backend.stream_calls[-1]["ref_codes"] is None
        engine.set_voice(QwenVoice("icl", ref_audio=wav, ref_text="Reference words."))
        assert engine.synthesize("Full reference cloning still works.")
        assert backend.stream_calls[-1]["ref_codes"] is not None
        assert backend.stream_calls[-1]["ref_text"] == "Reference words."
        assert calls[0]["cpu_threads"] == 6
        assert calls[0]["clamp_fp16"] is False
        assert engine.device == "cpu"
        assert engine.engine_name == "qwen_cpu"
        assert os.environ == before
    finally:
        engine.shutdown()
    assert backend.closed


def test_cpu_exports():
    from RealtimeTTS.engines import QwenCpuEngine as engine
    assert engine is QwenCpuEngine


def test_cpu_default_respects_small_host(tmp_path, monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 2)
    engine = QwenCpuEngine(
        backend_factory=lambda **_: FakeBackend(), warmup=False,
        voice_cache_dir=tmp_path / "cache",
    )
    assert engine.cpu_threads == 2
    assert engine.clone_mode == "speaker_only"
    engine.shutdown()
