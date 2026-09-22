import os

import pytest

from RealtimeTTS import QwenCpuEngine
from RealtimeTTS.qwen_emotions import build_parser, main
from tests.test_qwen_engine import FakeBackend


def test_cpu_overlap_options_are_per_context(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    before = dict(os.environ)
    calls = []
    backend = FakeBackend()

    def factory(**options):
        calls.append(options)
        return backend

    engine = QwenCpuEngine(
        cpu_threads=6, cpu_codec_threads=6, cpu_stream_frames=2,
        cpu_affinity=0x555, cpu_codec_affinity=0x555000,
        backend_factory=factory, warmup=False, voice_cache_dir=tmp_path,
    )
    try:
        assert engine.startup_buffer_ms == 160
        assert engine.quant == "Q8_0"
        for name, value in {
            "cpu_threads": 6, "cpu_codec_threads": 6, "cpu_stream_frames": 2,
            "cpu_affinity": 0x555, "cpu_codec_affinity": 0x555000,
        }.items():
            assert calls[0][name] == value
        assert os.environ == before
    finally:
        engine.shutdown()


@pytest.mark.parametrize("options,match", [
    ({"cpu_codec_threads": -1}, "cpu_codec_threads"),
    ({"cpu_codec_threads": True}, "cpu_codec_threads"),
    ({"cpu_stream_frames": 3}, "cpu_stream_frames"),
    ({"cpu_affinity": -1}, "cpu_affinity"),
    ({"cpu_affinity": 1 << 64}, "cpu_affinity"),
    ({"cpu_affinity": 1, "cpu_threads": 2}, "at least"),
    ({"cpu_codec_affinity": 3}, "requires cpu_codec_threads"),
    ({"cpu_codec_threads": 2, "cpu_codec_affinity": 1}, "at least"),
])
def test_cpu_invalid_scheduling_fails_before_loading(options, match):
    with pytest.raises(ValueError, match=match):
        QwenCpuEngine(**options)


def test_demo_parses_cpu_scheduling():
    args = build_parser().parse_args([
        "--device", "cpu", "--cpu-threads", "6", "--cpu-codec-threads", "6",
        "--cpu-stream-frames", "2", "--cpu-affinity", "0x555",
        "--cpu-codec-affinity", "0x555000",
    ])
    assert args.cpu_codec_threads == 6
    assert args.cpu_stream_frames == 2
    assert args.cpu_affinity == 0x555
    assert args.cpu_codec_affinity == 0x555000


@pytest.mark.parametrize("destination", [
    ["--device", "gpu"],
    ["--server", "http://localhost:8080"],
])
def test_demo_rejects_ignored_cpu_options(destination):
    with pytest.raises(SystemExit):
        main(destination + ["--cpu-codec-threads", "6"])
