"""ElevenlabsEngine must send its clarity/stability/style settings with each request."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("elevenlabs")
if "pyaudio" not in sys.modules:
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        stub = ModuleType("pyaudio")
        stub.paCustomFormat = 0
        sys.modules["pyaudio"] = stub

from RealtimeTTS.engines.elevenlabs_engine import ElevenlabsEngine  # noqa: E402


class _RecordingTextToSpeech:
    def __init__(self):
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return iter([b"audio"])


def test_synthesize_sends_configured_voice_settings():
    engine = ElevenlabsEngine(
        api_key="test", clarity=80.0, stability=30.0, style_exxageration=10.0
    )
    tts = _RecordingTextToSpeech()
    engine.client = SimpleNamespace(text_to_speech=tts)

    assert engine.synthesize("Hello there.")

    settings = tts.calls[0]["voice_settings"]
    assert settings.similarity_boost == pytest.approx(0.8)
    assert settings.stability == pytest.approx(0.3)
    assert settings.style == pytest.approx(0.1)


def test_set_voice_parameters_changes_the_next_request():
    engine = ElevenlabsEngine(api_key="test")
    tts = _RecordingTextToSpeech()
    engine.client = SimpleNamespace(text_to_speech=tts)

    engine.set_voice_parameters(stability=90.0)
    engine.synthesize("Hello there.")

    assert tts.calls[0]["voice_settings"].stability == pytest.approx(0.9)
