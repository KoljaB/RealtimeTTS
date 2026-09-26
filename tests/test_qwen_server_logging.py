"""Native diagnostics must not block ordinary server synthesis."""
import logging
import sys
from types import SimpleNamespace

import pytest
import RealtimeTTS.qwen_server as server


def test_native_diagnostics_are_debug_and_warnings_remain_visible(caplog):
    callbacks = []
    engine = SimpleNamespace(_backend=SimpleNamespace(set_log_callback=callbacks.append))
    server._configure_native_logging(engine)
    callback, = callbacks
    with caplog.at_level(logging.INFO, logger=server.LOGGER.name):
        for level, message in enumerate(("native debug", "native perf", "native warning", "native error")):
            callback(level, message)
    assert [(record.levelno, record.message) for record in caplog.records] == [
        (logging.WARNING, "native warning"), (logging.ERROR, "native error")
    ]
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=server.LOGGER.name):
        callback(1, "native perf")
    assert [(record.levelno, record.message) for record in caplog.records] == [(logging.DEBUG, "native perf")]


def test_logging_bridge_supports_backends_without_a_native_hook():
    server._configure_native_logging(object())
    server._configure_native_logging(SimpleNamespace(_backend=object()))


@pytest.mark.parametrize("device,engine_name", [("cpu", "QwenCpuEngine"), ("gpu", "QwenEngine")])
def test_cli_installs_logging_before_server_warmup(monkeypatch, device, engine_name):
    callbacks = []
    engine = SimpleNamespace(_backend=SimpleNamespace(set_log_callback=callbacks.append))
    monkeypatch.setattr(server, engine_name, lambda **kwargs: engine)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace())

    class ServerReached(Exception):
        pass

    def construct(actual_engine, **kwargs):
        assert actual_engine is engine
        assert len(callbacks) == 1
        raise ServerReached

    monkeypatch.setattr(server, "QwenHttpServer", construct)
    with pytest.raises(ServerReached):
        server.main(["--device", device])
