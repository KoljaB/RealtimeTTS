"""Breeze Studio API and native voice-design regressions."""
import base64
import json

import pytest

from RealtimeTTS import BreezeVoice
from RealtimeTTS.breeze_server import DESIGN_VOICE
from RealtimeTTS.engines.breeze_engine import BreezeEngineError
from tests.test_breeze_engine_server import Socket, make_engine, server_client, wav_bytes


def test_native_design_omits_reference_and_next_clone_restores_it(monkeypatch):
    socket = Socket()
    engine, http = make_engine(monkeypatch, socket)
    engine.set_voice(BreezeVoice("Designed", instruct="A warm alto voice.", voice_design=True))
    assert engine.synthesize("Hello (chuckle), welcome.")
    first = next(m for m in socket.messages if m["type"] == "start")
    assert first["voice_id"] == ""
    assert first["instruction"] == "A warm alto voice."
    assert not http.posts
    socket.incoming.put(json.dumps({"type": "ready"}))
    engine.set_voice(BreezeVoice("test", voice_id="test"))
    assert engine.synthesize("The reference voice is back.")
    starts = [m for m in socket.messages if m["type"] == "start"]
    assert starts[-1]["voice_id"] == "test"


def test_design_requires_description_and_excludes_reference(monkeypatch):
    engine, _ = make_engine(monkeypatch)
    with pytest.raises(BreezeEngineError, match="description"):
        engine.set_voice(BreezeVoice("Design", voice_design=True))
    with pytest.raises(BreezeEngineError, match="reference"):
        engine.set_voice(BreezeVoice("Design", voice_id="test", instruct="Warm.", voice_design=True))


def test_design_http_capabilities_validation_and_breeze_sampling(tmp_path):
    engine, client = server_client(tmp_path)
    with client:
        features = client.get("/v1/capabilities").json()["features"]
        assert features["voice_design"]
        assert features["voice_design_voice"] == DESIGN_VOICE
        assert "cfg_scale" in features["sampling_parameters"]
        assert "do_sample" not in features["sampling_parameters"]
        payload = {"voice": DESIGN_VOICE, "input": "Hello (laugh).", "language": "en"}
        assert client.post("/v1/audio/speech", json=payload).status_code == 400
        payload.update(instructions="A clear, warm voice.", cfg_scale=4, top_p=0, repetition_penalty=0)
        response = client.post("/v1/audio/speech", json=payload)
        assert response.status_code == 200 and response.content
        assert engine.voice.voice_design and not engine.voice.ref_audio
        assert engine.parameters["top_p"] == engine.parameters["repetition_penalty"] == 0
        for field, value in [("top_p", True), ("top_p", -1), ("top_p", 1.1), ("cfg_scale", 11)]:
            assert client.post("/v1/audio/speech", json={**payload, field:value}).status_code == 400
        assert client.post("/v1/audio/voices", json={"name": DESIGN_VOICE, "ref_text":"reference",
            "wav_b64":base64.b64encode(wav_bytes()).decode()}).status_code == 400
        assert client.post("/v1/audio/speech", json={"voice":"test", "input":"Reference again."}).status_code == 200
        assert not engine.voice.voice_design and engine.voice.ref_audio


def test_design_websocket_uses_same_streaming_protocol(tmp_path):
    engine, client = server_client(tmp_path)
    with client, client.websocket_connect("/v1/audio/speech-stream") as ws:
        ws.send_json({"type":"config", "voice":DESIGN_VOICE, "instructions":"A deep, gentle narrator.", "cfg_scale":4})
        ws.send_json({"type":"text", "text":"Hello (chuckle), come closer."})
        ws.send_json({"type":"end"})
        pcm = b""
        while True:
            event = ws.receive()
            if event.get("bytes"):
                pcm += event["bytes"]
            elif event.get("text"):
                payload = json.loads(event["text"])
                assert payload["type"] != "error", payload
                if payload["type"] == "done":
                    break
        assert pcm and engine.voice.voice_design
