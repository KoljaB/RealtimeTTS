import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location("guard_typed_test", Path(__file__).parents[1] / "tools/release_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def responses(monkeypatch, records):
    answers = iter([{"type": "o", "data": ["/org/freedesktop/systemd1/unit/test"]},
                    {"type": "a(sasbttttuii)", "data": records}])
    monkeypatch.setattr(guard.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(next(answers))))


def test_preserves_json_quotes_and_spaces(monkeypatch):
    argv = ["/venv/bin/python", "/venv/bin/server", "--options", '{"path":"a b","threads":7}']
    responses(monkeypatch, [["/venv/bin/python", argv, False, 0, 0, 0, 0, 0, 0, 0]])
    assert guard._systemd_typed_execstart_argv("test.service", argv[0]) == argv


@pytest.mark.parametrize("records", [[], [["/wrong/python", ["/wrong/python"]]], [["/venv/bin/python", [1]]]])
def test_rejects_missing_or_mismatched_command(monkeypatch, records):
    responses(monkeypatch, records)
    with pytest.raises(guard.GuardError):
        guard._systemd_typed_execstart_argv("test.service", "/venv/bin/python")
