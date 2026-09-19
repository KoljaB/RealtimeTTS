import sys
from types import SimpleNamespace

import pytest

import RealtimeTTS.qwen_server as server


def test_cpu_recovery_cli_is_opt_in_and_rejects_native_device():
    assert server.build_argument_parser().parse_args([]).onset_silence_recovery is False
    with pytest.raises(SystemExit):
        server.main(["--onset-silence-recovery"])


def test_cpu_recovery_cli_reaches_only_cpu_engine(monkeypatch):
    received = {}

    class EngineReached(Exception):
        pass

    def cpu_engine(**kwargs):
        received.update(kwargs)
        raise EngineReached

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace())
    monkeypatch.setattr(server, "QwenCpuEngine", cpu_engine)
    with pytest.raises(EngineReached):
        server.main([
            "--device", "cpu", "--onset-silence-recovery",
            "--onset-silence-profile", "qwen3_tts_12hz_0_6b_base_q8_v1",
        ])
    assert received["onset_silence_recovery"] is True
    assert received["onset_silence_profile"] == "qwen3_tts_12hz_0_6b_base_q8_v1"
    assert received["trim_silence"] is True
