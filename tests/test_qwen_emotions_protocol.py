"""Client guards for the public emotional showcase."""
import io
import json
import urllib.request

import pytest

from RealtimeTTS import qwen_emotions as demo


@pytest.mark.parametrize("model_type", ["custom_voice", "voice_design"])
def test_server_rejects_non_base_before_registering_a_voice(monkeypatch, model_type):
    calls = []

    def request(url, key, payload=None):
        calls.append(url)
        return io.BytesIO(json.dumps({
            "engine": {"model_id": demo.MODEL_ID},
            "features": {"model_type": model_type, "voice_cloning": False},
        }).encode())

    monkeypatch.setattr(demo, "_request", request)
    args = demo.build_parser().parse_args(["--server", "http://localhost:8080"])
    with pytest.raises(ValueError, match="requires a Base model"):
        demo.run_server(args, [])
    assert calls == ["http://localhost:8080/v1/capabilities"]


def test_api_client_never_redirects_an_authorization_header(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            assert request.get_header("Authorization") == "Bearer test-only"
            assert timeout == 180
            return "opened"

    def build_opener(handler):
        request = urllib.request.Request("https://tts.example/v1/capabilities")
        assert handler().redirect_request(
            request, None, 302, "Found", {}, "https://elsewhere.example/"
        ) is None
        return Opener()

    monkeypatch.setattr(demo.urllib.request, "build_opener", build_opener)
    assert demo._request("https://tts.example/v1/capabilities", "test-only") == "opened"
