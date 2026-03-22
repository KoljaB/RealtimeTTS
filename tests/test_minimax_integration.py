"""
Integration tests for MiniMaxEngine.

These tests make real API calls to the MiniMax TTS API.
They are skipped unless MINIMAX_API_KEY is set in the environment.

Run with: MINIMAX_API_KEY=your_key pytest tests/test_minimax_integration.py -v
"""

import os
import sys
import queue
import types
import importlib
import pytest
from unittest.mock import MagicMock

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)


def _import_minimax_engine():
    """Import minimax_engine in isolation without triggering full RealtimeTTS init."""
    mock_mp = MagicMock()
    mock_mp.Event.return_value = MagicMock()

    if "torch" not in sys.modules or isinstance(sys.modules["torch"], MagicMock):
        mock_torch = MagicMock()
        mock_torch.multiprocessing = mock_mp
        sys.modules["torch"] = mock_torch
        sys.modules["torch.multiprocessing"] = mock_mp

    base_engine_mod = types.ModuleType("RealtimeTTS.engines.base_engine")

    class BaseEngine:
        def __init__(self):
            self.engine_name = "unknown"
            self.can_consume_generators = False
            self.queue = queue.Queue()
            self.timings = queue.Queue()
            self.on_audio_chunk = None
            self.on_playback_start = None
            self.stop_synthesis_event = MagicMock()
            self.stop_synthesis_event.is_set.return_value = False
            self.audio_duration = 0

        def synthesize(self, text):
            self.stop_synthesis_event.clear()

        def shutdown(self):
            pass

    base_engine_mod.BaseEngine = BaseEngine

    engines_pkg = types.ModuleType("RealtimeTTS.engines")
    pkg = types.ModuleType("RealtimeTTS")
    pkg.engines = engines_pkg
    engines_pkg.base_engine = base_engine_mod

    sys.modules.setdefault("RealtimeTTS", pkg)
    sys.modules["RealtimeTTS.engines"] = engines_pkg
    sys.modules["RealtimeTTS.engines.base_engine"] = base_engine_mod

    spec = importlib.util.spec_from_file_location(
        "RealtimeTTS.engines.minimax_engine",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "RealtimeTTS",
            "engines",
            "minimax_engine.py",
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["RealtimeTTS.engines.minimax_engine"] = mod
    spec.loader.exec_module(mod)
    engines_pkg.minimax_engine = mod

    return mod.MiniMaxEngine


MiniMaxEngine = _import_minimax_engine()


@pytest.fixture
def engine():
    """Create a real MiniMaxEngine instance."""
    eng = MiniMaxEngine(debug=True)
    eng.queue = queue.Queue()
    eng.stop_synthesis_event = MagicMock()
    eng.stop_synthesis_event.is_set.return_value = False
    return eng


class TestMiniMaxIntegration:
    def test_synthesize_real_api(self, engine):
        """Test real API synthesis produces audio data."""
        result = engine.synthesize("Hello, this is a test.")
        assert result is True
        assert not engine.queue.empty()

        audio_data = engine.queue.get()
        assert len(audio_data) > 0
        # MP3 files typically start with FF FB/F3/F2 or ID3
        assert audio_data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2") or audio_data[:3] == b"ID3"

    def test_synthesize_turbo_model(self, engine):
        """Test synthesis with the turbo model."""
        engine.model = "speech-2.8-turbo"
        result = engine.synthesize("Quick turbo test.")
        assert result is True
        assert not engine.queue.empty()

    def test_synthesize_different_voice(self, engine):
        """Test synthesis with a different voice preset."""
        engine.set_voice("Deep_Voice_Man")
        result = engine.synthesize("Testing a different voice.")
        assert result is True
        assert not engine.queue.empty()
