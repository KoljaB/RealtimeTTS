"""No CLI or preset may silently opt into restarting quiet synthesis."""
import sys
from types import SimpleNamespace

import pytest

import RealtimeTTS.qwen_server as server


@pytest.mark.parametrize("flag", ["--onset-silence-recovery", "--no-onset-silence-recovery"])
def test_removed_recovery_cli_is_rejected(flag):
    with pytest.raises(SystemExit):
        server.build_argument_parser().parse_args([flag])


def test_cpu_cli_preserves_trimming_without_a_retry_option(monkeypatch):
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
            "--device", "cpu",
            "--onset-silence-profile", "qwen3_tts_12hz_0_6b_base_q8_v1",
        ])
    assert "onset_silence_recovery" not in received
    assert received["onset_silence_profile"] == "qwen3_tts_12hz_0_6b_base_q8_v1"
    assert received["trim_silence"] is True
