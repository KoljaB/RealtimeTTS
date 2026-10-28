"""
Unit tests for SpeechifyEngine.

Run with: pytest tests/test_speechify_engine.py -v
"""

import importlib.util
import os
import queue
import sys
import types
from unittest.mock import MagicMock


def _import_speechify_engine():
    """Import speechify_engine in isolation, without the full RealtimeTTS package."""
    names = [
        "pyaudio",
        "RealtimeTTS",
        "RealtimeTTS.engines",
        "RealtimeTTS.engines.base_engine",
        "RealtimeTTS.engines.speechify_engine",
    ]
    saved = {name: sys.modules.get(name) for name in names}

    if saved["pyaudio"] is None:
        pyaudio_stub = types.ModuleType("pyaudio")
        pyaudio_stub.paInt16 = 8
        sys.modules["pyaudio"] = pyaudio_stub

    base_mod = types.ModuleType("RealtimeTTS.engines.base_engine")

    class BaseEngine:
        def __init__(self):
            pass

        def synthesize(self, text, sentence_count=0):
            pass

    base_mod.BaseEngine = BaseEngine
    engines_pkg = types.ModuleType("RealtimeTTS.engines")
    pkg = types.ModuleType("RealtimeTTS")
    sys.modules["RealtimeTTS"] = pkg
    sys.modules["RealtimeTTS.engines"] = engines_pkg
    sys.modules["RealtimeTTS.engines.base_engine"] = base_mod

    spec = importlib.util.spec_from_file_location(
        "RealtimeTTS.engines.speechify_engine",
        os.path.join(os.path.dirname(__file__), "..", "RealtimeTTS", "engines", "speechify_engine.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["RealtimeTTS.engines.speechify_engine"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return mod.SpeechifyEngine


SpeechifyEngine = _import_speechify_engine()


def _engine(**kwargs):
    engine = SpeechifyEngine(api_key="test-key", **kwargs)
    # The real BaseEngine metaclass sets these up after __init__.
    engine.queue = queue.Queue()
    engine.stop_synthesis_event = MagicMock()
    engine.stop_synthesis_event.is_set.return_value = False
    response = MagicMock()
    response.status_code = 200
    response.iter_content.return_value = [b"\x00\x01\x02\x03"]
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    engine.session = MagicMock()
    engine.session.post.return_value = response
    return engine


def test_default_model_is_simba_3_5():
    assert SpeechifyEngine(api_key="test-key").model == "simba-3.5"


def test_default_request_uses_simba_3_5_without_language():
    engine = _engine()
    assert engine.synthesize("Hello there.") is True
    payload = engine.session.post.call_args.kwargs["json"]
    assert payload["model"] == "simba-3.5"
    assert payload["voice_id"] == "geffen_32"
    assert payload["output_format"] == "pcm_24000"
    assert "language" not in payload


def test_language_keeps_the_default_model():
    engine = _engine(language="fr-FR")
    assert engine.synthesize("Bonjour.") is True
    payload = engine.session.post.call_args.kwargs["json"]
    assert payload["model"] == "simba-3.5"
    assert payload["language"] == "fr-FR"


def test_explicit_model_is_respected():
    engine = _engine(model="simba-3.2")
    assert engine.synthesize("Hello.") is True
    assert engine.session.post.call_args.kwargs["json"]["model"] == "simba-3.2"


def test_caller_header_is_kept():
    engine = SpeechifyEngine(api_key="test-key")
    assert engine.session.headers["Speechify-Caller"] == "realtimetts"
