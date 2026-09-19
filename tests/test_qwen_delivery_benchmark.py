import numpy as np
import pytest

from tools import benchmark_qwen_delivery as benchmark


def test_delivery_audit_detects_gap_after_fast_first_packet(monkeypatch):
    timestamps = iter([10.0, 10.1, 10.5, 10.6])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(timestamps))
    class Backend:
        sample_rate = 1000
        def stream(self, text):
            yield np.zeros(200)
            yield np.ones(200) * .1
    result = benchmark.measure(Backend(), "demo")
    assert result["first_pcm_ms"] == pytest.approx(100)
    assert result["first_active_packet_ms"] == pytest.approx(500)
    assert result["minimum_gapless_start_ms"] == pytest.approx(300)


def test_empty_pcm_is_failure():
    class Backend:
        sample_rate = 1000
        def stream(self, text):
            return iter([])
    with pytest.raises(RuntimeError, match="no PCM"):
        benchmark.measure(Backend(), "demo")
