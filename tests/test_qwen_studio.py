import base64
import pytest
from fastapi.testclient import TestClient
from tests.test_qwen_server import FakeEngine, _server, _persist_voice
from RealtimeTTS.qwen_server import create_app


def test_studio_assets_public_but_voice_data_and_inference_protected(tmp_path):
    _persist_voice(tmp_path)
    with TestClient(create_app(_server(tmp_path, api_key="test-secret"))) as client:
        for path, mime in [("/", "text/html"), ("/studio/studio.js", "javascript"), ("/studio/studio.css", "text/css")]:
            response = client.get(path)
            assert response.status_code == 200
            assert mime in response.headers["content-type"]
            assert "test-secret" not in response.text
        assert client.get("/v1/audio/voices").status_code == 401
        assert client.get("/v1/capabilities").status_code == 401
        assert client.post("/v1/audio/speech", json={"voice": "mira", "input": "hello"}).status_code == 401
        assert client.get("/studio/qwen_server.py", headers={"Authorization": "Bearer test-secret"}).status_code == 404


def test_browser_websocket_bearer_subprotocol_is_not_echoed(tmp_path):
    _persist_voice(tmp_path)
    token = base64.urlsafe_b64encode(b"test-secret").decode().rstrip("=")
    with TestClient(create_app(_server(tmp_path, api_key="test-secret"))) as client:
        with client.websocket_connect("/v1/audio/speech-stream", subprotocols=["qwen-studio", "bearer." + token]) as ws:
            assert ws.accepted_subprotocol == "qwen-studio"
            ws.send_json({"type": "config", "voice": "mira"})
            ws.send_json({"type": "text", "text": "hello."})
            ws.send_json({"type": "end"})
            received = []
            while True:
                event = ws.receive()
                if event.get("bytes"):
                    received.append(event["bytes"])
                if event.get("text") and '"type":"done"' in event["text"]:
                    break
            assert received


def test_codec_sampling_validation_and_defaults_reset(tmp_path):
    engine = FakeEngine()
    engine.do_sample = True
    engine.subtalker_do_sample = True
    engine.subtalker_top_k = 50
    engine.subtalker_top_p = 1.0
    engine.subtalker_temperature = 0.9
    _persist_voice(tmp_path)
    with TestClient(create_app(_server(tmp_path, engine))) as client:
        payload = {"voice": "mira", "input": "hello", "do_sample": False,
                   "subtalker_do_sample": False, "subtalker_top_k": 12,
                   "subtalker_top_p": .7, "subtalker_temperature": .5}
        assert client.post("/v1/audio/speech", json=payload).status_code == 200
        assert engine.parameters["subtalker_top_k"] == 12
        assert client.post("/v1/audio/speech", json={"voice": "mira", "input": "hello"}).status_code == 200
        assert engine.parameters["subtalker_top_k"] == 50
        assert engine.parameters["do_sample"] is True
        for field, value in [("do_sample", 1), ("subtalker_top_k", -1), ("subtalker_top_p", 0), ("model", "not-loaded")]:
            assert client.post("/v1/audio/speech", json={**payload, field: value}).status_code == 400


@pytest.mark.parametrize("mode,instruction_control", [("custom_voice", True), ("voice_design", True), ("custom_voice", False)])
def test_nonreference_model_contract(tmp_path, mode, instruction_control):
    class ModeEngine(FakeEngine):
        def set_voice(self, voice):
            self.current_voice = voice
    engine = ModeEngine()
    engine.model_type = mode
    engine.instruction_control = instruction_control
    engine.built_in_speakers = ("Ryan", "Vivian") if mode == "custom_voice" else ()
    with TestClient(create_app(_server(tmp_path, engine))) as client:
        voices = client.get("/v1/audio/voices").json()["voices"]
        voice = voices[0]["name"]
        payload = {"voice": voice, "input": "hello"}
        if instruction_control:
            payload["instructions"] = "A calm, clear voice."
        assert client.post("/v1/audio/speech", json=payload).status_code == 200
        assert engine.current_voice.speaker == (voice if mode == "custom_voice" else None)
        assert not engine.current_voice.ref_audio
        if mode == "voice_design":
            assert client.post("/v1/audio/speech", json={"input": "hello"}).status_code == 400
        if not instruction_control:
            assert client.post("/v1/audio/speech", json={**payload, "instructions": "angry"}).status_code == 400
        assert client.post("/v1/audio/voices", json={}).status_code == 400
