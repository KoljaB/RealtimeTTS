"""The standalone server must translate Studio language codes for the native engine."""
import pytest
from fastapi.testclient import TestClient

from tests.test_qwen_server import _persist_voice, _server
from RealtimeTTS.qwen_server import create_app

@pytest.mark.parametrize("language, expected", [
    ("de", "german"), ("en", "english"), ("es", "spanish"),
    ("fr", "french"), ("it", "italian"), ("pt", "portuguese"),
    ("ru", "russian"), ("zh", "chinese"), ("ja", "japanese"),
    ("ko", "korean"), ("auto", "auto"), (" DE_de ", "german"),
    ("German", "german"),
])
def test_http_language_codes_reach_engine_as_native_names(tmp_path, language, expected):
    _persist_voice(tmp_path)
    server = _server(tmp_path)
    assert server.language_router is None
    with TestClient(create_app(server)) as client:
        response = client.post("/v1/audio/speech", json={
            "input": "audible", "voice": "mira",
            "language": language, "response_format": "wav",
        })
    assert response.status_code == 200
    assert response.content.startswith(b"RIFF")
    assert server.engine.current_voice.language == expected

@pytest.mark.parametrize("language, expected", [("de", "german"), ("en", "english"), ("Auto", "auto")])
def test_startup_warmup_normalizes_default_language(tmp_path, language, expected):
    _persist_voice(tmp_path)
    server = _server(tmp_path, language=language, startup_warmup_voice="mira")
    with TestClient(create_app(server)):
        assert server.engine.warmup_calls == 1
        assert server.engine.current_voice.language == expected


@pytest.mark.parametrize("language, expected", [("de", "german"), ("en", "english"), ("Auto", "auto")])
def test_explicit_warmup_normalizes_language(tmp_path, language, expected):
    _persist_voice(tmp_path)
    server = _server(tmp_path)
    server.warmup_voice("mira", language=language)
    assert server.engine.warmup_calls == 1
    assert server.engine.current_voice.language == expected


def test_invalid_warmup_language_is_rejected_before_native_call(tmp_path):
    _persist_voice(tmp_path)
    server = _server(tmp_path)
    with pytest.raises(ValueError, match="unsupported Qwen language"):
        server.warmup_voice("mira", language="zz")
    assert server.engine.warmup_calls == 0


def test_unsupported_language_is_rejected_before_native_synthesis(tmp_path):
    _persist_voice(tmp_path)
    server = _server(tmp_path)
    with TestClient(create_app(server)) as client:
        response = client.post("/v1/audio/speech", json={
            "input": "audible", "voice": "mira", "language": "zz",
        })
    assert response.status_code == 400
    assert "unsupported Qwen language" in response.json()["error"]["message"]
    assert server.engine.current_voice is None
