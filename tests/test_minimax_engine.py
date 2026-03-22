"""
Unit tests for MiniMaxEngine.

Run with: pytest tests/test_minimax_engine.py -v
"""

import os
import json
import queue
import sys
import importlib
import types
import pytest
from unittest.mock import patch, MagicMock

# --- Isolated import of minimax_engine to avoid heavy RealtimeTTS deps ---
# We mock torch.multiprocessing and base_engine to avoid importing the full package


def _import_minimax_engine():
    """Import minimax_engine in isolation without triggering full RealtimeTTS init."""
    # Mock torch.multiprocessing
    mock_mp = MagicMock()
    mock_mp.Event.return_value = MagicMock()

    saved_modules = {}

    # Save and mock required modules
    for mod_name in ["torch", "torch.multiprocessing"]:
        saved_modules[mod_name] = sys.modules.get(mod_name)

    mock_torch = MagicMock()
    mock_torch.multiprocessing = mock_mp
    sys.modules["torch"] = mock_torch
    sys.modules["torch.multiprocessing"] = mock_mp

    # Create a minimal base_engine module
    import numpy as np

    base_engine_mod = types.ModuleType("RealtimeTTS.engines.base_engine")

    class TimingInfo:
        def __init__(self, start_time, end_time, word):
            self.start_time = start_time
            self.end_time = end_time
            self.word = word

    class BaseEngine:
        def __init__(self):
            self.engine_name = "unknown"
            self.can_consume_generators = False
            self.queue = queue.Queue()
            self.timings = queue.Queue()
            self.on_audio_chunk = None
            self.on_playback_start = None
            self.stop_synthesis_event = MagicMock()
            self.audio_duration = 0

        def synthesize(self, text):
            self.stop_synthesis_event.clear()

        def shutdown(self):
            pass

    base_engine_mod.BaseEngine = BaseEngine
    base_engine_mod.TimingInfo = TimingInfo

    # Temporarily set module
    engines_pkg = types.ModuleType("RealtimeTTS.engines")
    pkg = types.ModuleType("RealtimeTTS")

    saved_modules["RealtimeTTS"] = sys.modules.get("RealtimeTTS")
    saved_modules["RealtimeTTS.engines"] = sys.modules.get("RealtimeTTS.engines")
    saved_modules["RealtimeTTS.engines.base_engine"] = sys.modules.get(
        "RealtimeTTS.engines.base_engine"
    )

    pkg.engines = engines_pkg
    engines_pkg.base_engine = base_engine_mod
    sys.modules["RealtimeTTS"] = pkg
    sys.modules["RealtimeTTS.engines"] = engines_pkg
    sys.modules["RealtimeTTS.engines.base_engine"] = base_engine_mod

    # Now load the minimax engine
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

    # Wire up module references for @patch to resolve
    engines_pkg.minimax_engine = mod
    pkg.engines = engines_pkg

    return mod.MiniMaxEngine, mod.MiniMaxVoice


MiniMaxEngine, MiniMaxVoice = _import_minimax_engine()


# --- Fixtures ---


@pytest.fixture
def mock_env_key():
    """Set MINIMAX_API_KEY env var for testing."""
    with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-api-key-123"}):
        yield


@pytest.fixture
def engine(mock_env_key):
    """Create a MiniMaxEngine instance."""
    eng = MiniMaxEngine()
    eng.queue = queue.Queue()
    eng.stop_synthesis_event = MagicMock()
    eng.stop_synthesis_event.is_set.return_value = False
    return eng


# --- Unit Tests: Initialization ---


class TestMiniMaxEngineInit:
    def test_init_with_api_key_param(self):
        """Test initialization with explicit API key."""
        eng = MiniMaxEngine(api_key="explicit-key")
        assert eng.api_key == "explicit-key"

    def test_init_with_env_var(self, mock_env_key):
        """Test initialization with MINIMAX_API_KEY env var."""
        eng = MiniMaxEngine()
        assert eng.api_key == "test-api-key-123"

    def test_init_raises_without_api_key(self):
        """Test that ValueError is raised when no API key is provided."""
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="MiniMax API key is required"):
                MiniMaxEngine()

    def test_init_default_params(self, engine):
        """Test default parameter values."""
        assert engine.model == "speech-2.8-hd"
        assert engine.voice == "English_Graceful_Lady"
        assert engine.speed == 1.0
        assert engine.volume == 1.0
        assert engine.pitch == 0
        assert engine.debug is False

    def test_init_custom_params(self, mock_env_key):
        """Test initialization with custom parameters."""
        eng = MiniMaxEngine(
            model="speech-2.8-turbo",
            voice="Deep_Voice_Man",
            speed=1.5,
            volume=2.0,
            pitch=3,
            debug=True,
        )
        assert eng.model == "speech-2.8-turbo"
        assert eng.voice == "Deep_Voice_Man"
        assert eng.speed == 1.5
        assert eng.volume == 2.0
        assert eng.pitch == 3
        assert eng.debug is True

    def test_post_init_sets_engine_name(self, engine):
        """Test that post_init sets the engine name."""
        engine.post_init()
        assert engine.engine_name == "minimax"

    def test_base_url(self, engine):
        """Test the API base URL."""
        assert engine.base_url == "https://api.minimax.io/v1/t2a_v2"


# --- Unit Tests: Stream Info ---


class TestMiniMaxStreamInfo:
    def test_get_stream_info_format(self, engine):
        """Test stream info returns custom format for MP3."""
        import pyaudio

        fmt, channels, sample_rate = engine.get_stream_info()
        assert fmt == pyaudio.paCustomFormat

    def test_get_stream_info_channels(self, engine):
        """Test stream info returns mono."""
        _, channels, _ = engine.get_stream_info()
        assert channels == 1

    def test_get_stream_info_sample_rate(self, engine):
        """Test stream info returns 32000 Hz."""
        _, _, sample_rate = engine.get_stream_info()
        assert sample_rate == 32000


# --- Unit Tests: Voice Management ---


class TestMiniMaxVoiceManagement:
    def test_get_voices_returns_all(self, engine):
        """Test that get_voices returns all 12 voice presets."""
        voices = engine.get_voices()
        assert len(voices) == 12
        assert all(isinstance(v, MiniMaxVoice) for v in voices)

    def test_get_voices_has_english_voices(self, engine):
        """Test that English voices are present."""
        voices = engine.get_voices()
        names = [v.name for v in voices]
        assert "English_Graceful_Lady" in names
        assert "English_Persuasive_Man" in names
        assert "English_radiant_girl" in names
        assert "English_Insightful_Speaker" in names
        assert "English_Lucky_Robot" in names

    def test_get_voices_has_multilingual_voices(self, engine):
        """Test that multilingual voices are present."""
        voices = engine.get_voices()
        names = [v.name for v in voices]
        assert "Deep_Voice_Man" in names
        assert "sweet_girl" in names
        assert "Wise_Woman" in names

    def test_set_voice_by_name(self, engine):
        """Test setting voice by string name."""
        engine.set_voice("Deep_Voice_Man")
        assert engine.voice == "Deep_Voice_Man"

    def test_set_voice_case_insensitive(self, engine):
        """Test setting voice with case-insensitive matching."""
        engine.set_voice("deep_voice_man")
        assert engine.voice == "Deep_Voice_Man"

    def test_set_voice_by_object(self, engine):
        """Test setting voice by MiniMaxVoice object."""
        voice = MiniMaxVoice("English_Persuasive_Man")
        engine.set_voice(voice)
        assert engine.voice == "English_Persuasive_Man"

    def test_set_voice_arbitrary_id(self, engine):
        """Test setting an arbitrary voice ID."""
        engine.set_voice("custom_voice_id")
        assert engine.voice == "custom_voice_id"

    def test_set_voice_parameters_speed(self, engine):
        """Test setting speed parameter."""
        engine.set_voice_parameters(speed=1.5)
        assert engine.speed == 1.5

    def test_set_voice_parameters_volume(self, engine):
        """Test setting volume parameter."""
        engine.set_voice_parameters(volume=2.0)
        assert engine.volume == 2.0

    def test_set_voice_parameters_pitch(self, engine):
        """Test setting pitch parameter."""
        engine.set_voice_parameters(pitch=5)
        assert engine.pitch == 5

    def test_set_voice_parameters_model(self, engine):
        """Test setting model parameter."""
        engine.set_voice_parameters(model="speech-2.8-turbo")
        assert engine.model == "speech-2.8-turbo"

    def test_set_voice_parameters_multiple(self, engine):
        """Test setting multiple parameters at once."""
        engine.set_voice_parameters(speed=1.5, volume=2.0, pitch=5, model="speech-2.8-turbo")
        assert engine.speed == 1.5
        assert engine.volume == 2.0
        assert engine.pitch == 5
        assert engine.model == "speech-2.8-turbo"


# --- Unit Tests: MiniMaxVoice ---


class TestMiniMaxVoice:
    def test_voice_creation(self):
        """Test MiniMaxVoice object creation."""
        voice = MiniMaxVoice("test_voice", language="english")
        assert voice.name == "test_voice"
        assert voice.language == "english"

    def test_voice_repr(self):
        """Test MiniMaxVoice string representation."""
        voice = MiniMaxVoice("English_Graceful_Lady")
        assert repr(voice) == "English_Graceful_Lady"

    def test_voice_default_language(self):
        """Test MiniMaxVoice default language."""
        voice = MiniMaxVoice("test")
        assert voice.language == "english"


# --- Unit Tests: Synthesize ---


class TestMiniMaxSynthesize:
    def _make_success_response(self, audio_hex="48656c6c6f"):
        """Create a mock successful API response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"audio": audio_hex},
        }
        return mock_resp

    def _make_error_response(self, status_code=1000, status_msg="Error"):
        """Create a mock error API response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "base_resp": {"status_code": status_code, "status_msg": status_msg},
            "data": {},
        }
        return mock_resp

    @patch("RealtimeTTS.engines.minimax_engine.requests.post")
    def test_synthesize_success(self, mock_post, engine):
        """Test successful synthesis."""
        audio_hex = "ff" * 100
        mock_post.return_value = self._make_success_response(audio_hex)

        result = engine.synthesize("Hello world")

        assert result is True
        assert not engine.queue.empty()
        audio_data = engine.queue.get()
        assert len(audio_data) == 100

    @patch("RealtimeTTS.engines.minimax_engine.requests.post")
    def test_synthesize_sends_correct_payload(self, mock_post, engine):
        """Test that synthesize sends the correct API payload."""
        mock_post.return_value = self._make_success_response()

        engine.synthesize("Test text")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")

        assert payload["model"] == "speech-2.8-hd"
        assert payload["text"] == "Test text"
        assert payload["voice_setting"]["voice_id"] == "English_Graceful_Lady"
        assert payload["voice_setting"]["speed"] == 1.0
        assert payload["voice_setting"]["vol"] == 1.0
        assert payload["voice_setting"]["pitch"] == 0
        assert payload["audio_setting"]["format"] == "mp3"

    @patch("RealtimeTTS.engines.minimax_engine.requests.post")
    def test_synthesize_sends_auth_header(self, mock_post, engine):
        """Test that synthesize includes authorization header."""
        mock_post.return_value = self._make_success_response()

        engine.synthesize("Test")

        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer test-api-key-123"
        assert headers["Content-Type"] == "application/json"

    @patch("RealtimeTTS.engines.minimax_engine.requests.post")
    def test_synthesize_api_error(self, mock_post, engine):
        """Test handling of API error response."""
        mock_post.return_value = self._make_error_response(1000, "Invalid API key")

        result = engine.synthesize("Hello")

        assert result is False
        assert engine.queue.empty()

    @patch("RealtimeTTS.engines.minimax_engine.requests.post")
    def test_synthesize_empty_audio(self, mock_post, engine):
        """Test handling of empty audio response."""
        mock_post.return_value = self._make_success_response("")

        result = engine.synthesize("Hello")

        assert result is False
        assert engine.queue.empty()

    @patch("RealtimeTTS.engines.minimax_engine.requests.post")
    def test_synthesize_request_exception(self, mock_post, engine):
        """Test handling of network request failure."""
        import requests as req

        mock_post.side_effect = req.exceptions.ConnectionError("Connection refused")

        result = engine.synthesize("Hello")

        assert result is False
        assert engine.queue.empty()

    @patch("RealtimeTTS.engines.minimax_engine.requests.post")
    def test_synthesize_invalid_hex(self, mock_post, engine):
        """Test handling of invalid hex audio data."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"audio": "not-valid-hex!!!"},
        }
        mock_post.return_value = mock_resp

        result = engine.synthesize("Hello")

        assert result is False

    @patch("RealtimeTTS.engines.minimax_engine.requests.post")
    def test_synthesize_stop_event(self, mock_post, engine):
        """Test that synthesis respects stop event."""
        mock_post.return_value = self._make_success_response("ff" * 50)
        engine.stop_synthesis_event.is_set.return_value = True

        result = engine.synthesize("Hello")

        assert result is False

    @patch("RealtimeTTS.engines.minimax_engine.requests.post")
    def test_synthesize_with_turbo_model(self, mock_post, engine):
        """Test synthesis with turbo model."""
        engine.model = "speech-2.8-turbo"
        mock_post.return_value = self._make_success_response("ff" * 50)

        result = engine.synthesize("Fast speech")

        assert result is True
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["model"] == "speech-2.8-turbo"

    @patch("RealtimeTTS.engines.minimax_engine.requests.post")
    def test_synthesize_custom_voice_params(self, mock_post, engine):
        """Test synthesis with custom voice parameters."""
        engine.speed = 1.5
        engine.volume = 0.8
        engine.pitch = -3
        mock_post.return_value = self._make_success_response("ff" * 50)

        engine.synthesize("Custom params")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["voice_setting"]["speed"] == 1.5
        assert payload["voice_setting"]["vol"] == 0.8
        assert payload["voice_setting"]["pitch"] == -3

    @patch("RealtimeTTS.engines.minimax_engine.requests.post")
    def test_synthesize_http_error(self, mock_post, engine):
        """Test handling of HTTP error status."""
        import requests as req

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req.exceptions.HTTPError("500 Server Error")
        mock_post.return_value = mock_resp

        result = engine.synthesize("Hello")

        assert result is False

    @patch("RealtimeTTS.engines.minimax_engine.requests.post")
    def test_synthesize_timeout(self, mock_post, engine):
        """Test handling of request timeout."""
        import requests as req

        mock_post.side_effect = req.exceptions.Timeout("Request timed out")

        result = engine.synthesize("Hello")

        assert result is False


# --- Unit Tests: Shutdown ---


class TestMiniMaxShutdown:
    def test_shutdown(self, engine):
        """Test shutdown doesn't raise."""
        engine.shutdown()  # Should not raise
