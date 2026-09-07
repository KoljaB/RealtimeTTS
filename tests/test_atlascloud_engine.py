import importlib.util
import os
import queue
import sys
import types
from unittest.mock import MagicMock, call, patch


def _import_engine():
    saved = dict(sys.modules)
    base_module = types.ModuleType("RealtimeTTS.engines.base_engine")

    class BaseEngine:
        def __init__(self):
            self.queue = queue.Queue()
            self.stop_synthesis_event = MagicMock()

        def synthesize(self, text, sentence_count=0):
            del text, sentence_count

    base_module.BaseEngine = BaseEngine
    package = types.ModuleType("RealtimeTTS")
    engines = types.ModuleType("RealtimeTTS.engines")
    pyaudio = types.ModuleType("pyaudio")
    pyaudio.paCustomFormat = 8
    sys.modules.update(
        {
            "RealtimeTTS": package,
            "RealtimeTTS.engines": engines,
            "RealtimeTTS.engines.base_engine": base_module,
            "pyaudio": pyaudio,
        }
    )
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "RealtimeTTS",
        "engines",
        "atlascloud_engine.py",
    )
    spec = importlib.util.spec_from_file_location(
        "RealtimeTTS.engines.atlascloud_engine", path
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.clear()
        sys.modules.update(saved)
    return module


MODULE = _import_engine()
AtlasCloudEngine = MODULE.AtlasCloudEngine


def test_requires_api_key():
    with patch.dict(os.environ, {}, clear=True):
        try:
            AtlasCloudEngine()
        except ValueError as exc:
            assert "Atlas Cloud API key is required" in str(exc)
        else:
            raise AssertionError("expected missing API key to fail")


def test_submits_once_polls_and_queues_audio():
    engine = AtlasCloudEngine(api_key="test", poll_interval=0)
    engine.queue = queue.Queue()
    engine.stop_synthesis_event = MagicMock()
    engine.stop_synthesis_event.is_set.return_value = False

    submit = MagicMock()
    submit.raise_for_status.return_value = None
    submit.json.return_value = {"data": {"id": "prediction-1"}}
    pending = MagicMock()
    pending.raise_for_status.return_value = None
    pending.json.return_value = {"data": {"status": "processing"}}
    complete = MagicMock()
    complete.raise_for_status.return_value = None
    complete.json.return_value = {
        "data": {"status": "completed", "outputs": ["https://audio.test/a.mp3"]}
    }
    audio = MagicMock()
    audio.raise_for_status.return_value = None
    audio.iter_content.return_value = [b"one", b"two"]

    with (
        patch.object(MODULE.requests, "post", return_value=submit) as post,
        patch.object(
            MODULE.requests, "get", side_effect=[pending, complete, audio]
        ) as get,
    ):
        assert engine.synthesize("hello") is True

    post.assert_called_once()
    assert get.call_args_list == [
        call(
            "https://api.atlascloud.ai/api/v1/model/prediction/prediction-1",
            headers=engine._headers(),
            timeout=60,
        ),
        call(
            "https://api.atlascloud.ai/api/v1/model/prediction/prediction-1",
            headers=engine._headers(),
            timeout=60,
        ),
        call("https://audio.test/a.mp3", stream=True, timeout=60),
    ]
    assert engine.queue.get_nowait() == b"one"
    assert engine.queue.get_nowait() == b"two"
