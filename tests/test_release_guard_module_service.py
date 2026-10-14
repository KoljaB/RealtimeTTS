import hashlib
import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


GUARD_PATH = Path(__file__).parents[1] / "tools" / "release_guard.py"
SPEC = importlib.util.spec_from_file_location("release_guard_module_service", GUARD_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import setup failure
    raise RuntimeError(f"cannot load release guard from {GUARD_PATH}")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class _FakePath:
    command_bytes = b""
    environment_bytes = b""

    def __init__(self, value):
        self.value = str(value)

    def __truediv__(self, part):
        return type(self)(self.value.rstrip("/") + "/" + str(part))

    def __str__(self):
        return self.value

    def __repr__(self):  # pragma: no cover - assertion aid
        return f"_FakePath({self.value!r})"

    def read_bytes(self):
        if self.value.endswith("/cmdline"):
            return type(self).command_bytes
        if self.value.endswith("/environ"):
            return type(self).environment_bytes
        raise AssertionError(f"unexpected read: {self.value}")

    def is_file(self):
        return True

    def resolve(self):
        return self


def _raw_argv(argv):
    return b"\0".join(item.encode() for item in argv) + b"\0"


def _service_state(
    monkeypatch,
    process_argv,
    unit_argv,
    *,
    module_origin="/venv/lib/python3.12/site-packages/RealtimeTTS/qwen_server.py",
):
    _FakePath.command_bytes = _raw_argv(process_argv)
    _FakePath.environment_bytes = b"LANG=C\0"
    monkeypatch.setattr(guard, "Path", _FakePath)
    real_os = guard.os
    monkeypatch.setattr(
        guard,
        "os",
        SimpleNamespace(
            name="posix",
            path=real_os.path,
            readlink=lambda path: "/srv/realtimetts",
        ),
    )
    monkeypatch.setattr(
        guard,
        "_sha256_file",
        lambda path: hashlib.sha256(str(path).encode()).hexdigest(),
    )
    output = "\n".join(
        [
            "ActiveState=active",
            "SubState=running",
            "MainPID=4242",
            (
                "ExecStart={ path="
                + unit_argv[0]
                + " argv[]="
                + shlex.join(unit_argv)
                + " ; }"
            ),
            "WorkingDirectory=/srv/realtimetts",
            "FragmentPath=/etc/systemd/user/wwz-qwen3-tts-cpu.service",
        ]
    )
    probe_output = json.dumps(
        {"module": "RealtimeTTS.qwen_server", "origin": module_origin}
    )

    def run(command, *args, **kwargs):
        if command[0] == "systemctl":
            return SimpleNamespace(returncode=0, stdout=output, stderr="")
        assert kwargs["cwd"] == "/srv/realtimetts"
        assert kwargs["env"]["LANG"] == "C"
        return SimpleNamespace(returncode=0, stdout=probe_output, stderr="")

    monkeypatch.setattr(guard.subprocess, "run", run)
    monkeypatch.setattr(
        guard,
        "_systemd_typed_execstart_argv",
        lambda service, launcher: unit_argv,
    )
    return guard._systemd_user_service_state(
        "wwz-qwen3-tts-cpu.service",
        {
            "python": "/venv/bin/python",
            "prefix": "/venv",
            "imports": {
                "RealtimeTTS": "/venv/lib/python3.12/site-packages/RealtimeTTS"
            },
        },
    )


def test_accepts_attested_python_module_service(monkeypatch):
    argv = [
        "/venv/bin/python",
        "-m",
        "RealtimeTTS.qwen_server",
        "--port",
        "18084",
    ]
    state = _service_state(monkeypatch, argv, argv)

    assert state["launcher"] == "/venv/bin/python"
    assert state["launcher_sha256"] == hashlib.sha256(
        b"/venv/bin/python"
    ).hexdigest()
    assert state["command_sha256"] == hashlib.sha256(_raw_argv(argv)).hexdigest()
    assert state["module"] == "RealtimeTTS.qwen_server"
    assert (
        state["module_origin"]
        == "/venv/lib/python3.12/site-packages/RealtimeTTS/qwen_server.py"
    )
    assert state["module_origin_sha256"] == hashlib.sha256(
        state["module_origin"].encode()
    ).hexdigest()


def test_rejects_module_outside_attested_runtime(monkeypatch):
    argv = ["/venv/bin/python", "-m", "untrusted_server"]

    with pytest.raises(guard.GuardError, match="outside the attested runtime"):
        _service_state(monkeypatch, argv, argv)


def test_module_service_keeps_process_and_unit_argv_in_lockstep(monkeypatch):
    unit_argv = ["/venv/bin/python", "-m", "RealtimeTTS.qwen_server", "--port", "18084"]
    process_argv = ["/venv/bin/python", "-m", "RealtimeTTS.qwen_server", "--port", "18085"]

    with pytest.raises(guard.GuardError, match="Python arguments differ from its unit"):
        _service_state(monkeypatch, process_argv, unit_argv)


def test_rejects_cwd_shadowed_module(tmp_path):
    shadow_root = tmp_path / "cwd"
    shadow_package = shadow_root / "RealtimeTTS"
    shadow_package.mkdir(parents=True)
    (shadow_package / "__init__.py").write_text("", encoding="utf-8")
    (shadow_package / "qwen_server.py").write_text(
        "SHADOW = True\n", encoding="utf-8"
    )

    trusted_root = tmp_path / "site-packages" / "RealtimeTTS"
    trusted_root.mkdir(parents=True)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    with pytest.raises(
        guard.GuardError, match="module origin is outside the attested runtime package"
    ):
        guard._resolve_service_module(
            "wwz-qwen3-tts-cpu.service",
            sys.executable,
            str(shadow_root),
            environment,
            "RealtimeTTS.qwen_server",
            {"imports": {"RealtimeTTS": str(trusted_root)}},
        )