import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from RealtimeTTS import breeze_worker


@pytest.mark.parametrize("failure", ["none", "body", "timeout", "exit", "health", "cpu", "kill"])
def test_owned_worker_cleanup_and_readiness(monkeypatch, tmp_path, failure):
    model = tmp_path / "voice q8_0.gguf"
    model.write_bytes(b"GGUF")
    state = {"terminated": False, "killed": False, "command": None}

    class Process:
        returncode = 7 if failure == "exit" else None
        def poll(self):
            return self.returncode
        def terminate(self):
            state["terminated"] = True
        def wait(self, timeout=None):
            if failure == "kill" and not state["killed"]:
                raise subprocess.TimeoutExpired("worker", timeout)
            self.returncode = 0
        def kill(self):
            state["killed"] = True

    process = Process()
    def launch(command, **kwargs):
        state["command"] = command
        return process

    class Session:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, url, timeout):
            assert self.trust_env is False
            command = state["command"]
            port = int(command[command.index("--ws-port") + 1])
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"native_checkpoints": failure != "health",
                              "sample_rate": 24000, "ws_port": port,
                              "backend": "CPU" if failure == "cpu" else "Vulkan0"},
            )

    monkeypatch.setitem(sys.modules, "realtimetts_breeze_native",
                        SimpleNamespace(executable_path=lambda: Path("breeze-server")))
    monkeypatch.setattr(breeze_worker.subprocess, "Popen", launch)
    monkeypatch.setattr(breeze_worker.requests, "Session", Session)

    def run():
        with breeze_worker.local_worker(model, tmp_path / "voices",
                                        timeout=-1 if failure == "timeout" else 1) as urls:
            assert urls[0].startswith("http://127.0.0.1:")
            assert urls[1].startswith("ws://127.0.0.1:")
            assert urls[0].rsplit(":", 1)[1] != urls[1].rsplit(":", 1)[1]
            assert not state["terminated"]
            if failure == "body":
                raise ValueError("API startup failed")

    if failure in ("body", "timeout", "exit", "health", "cpu"):
        error = {"body": ValueError, "timeout": TimeoutError}.get(failure, RuntimeError)
        with pytest.raises(error):
            run()
    else:
        run()
    assert state["terminated"] is (failure != "exit")
    assert state["killed"] is (failure == "kill")
    assert state["command"][1] == str(model.resolve())
