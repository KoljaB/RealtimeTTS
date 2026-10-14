"""Quiet speech is valid output, independently of its loudness."""
import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from RealtimeTTS.qwen_server import create_app
from tests.test_qwen_server import FakeEngine, _register, _server


@pytest.mark.parametrize("response_format", ["pcm", "wav"])
@pytest.mark.parametrize("amplitude", [1, 10, 79])
def test_quiet_pcm_is_preserved_and_not_reported_as_failure(tmp_path, response_format, amplitude):
    engine = FakeEngine()
    engine.audible = np.array([amplitude, -amplitude] * 120, dtype="<i2").tobytes()
    server = _server(tmp_path, engine)
    with TestClient(create_app(server)) as client:
        _register(client)
        response = client.post("/v1/audio/speech", json={
            "input": "A quiet whisper.", "voice": "mira", "response_format": response_format})
        assert response.status_code == 200, response.text
        actual = response.content
        if response_format == "wav":
            with wave.open(io.BytesIO(actual)) as audio:
                actual = audio.readframes(audio.getnframes())
        assert actual == engine.silent + engine.audible
        assert client.get("/health").json()["unusable_outputs_total"] == 0


def test_websocket_quiet_pcm_finishes_without_error(tmp_path):
    engine = FakeEngine()
    engine.audible = np.array([1, -1] * 120, dtype="<i2").tobytes()
    with TestClient(create_app(_server(tmp_path, engine, sentence_silence_duration=0.0,
                                            comma_silence_duration=0.0))) as client:
        _register(client)
        with client.websocket_connect("/v1/audio/speech-stream") as websocket:
            websocket.send_json({"type": "config", "voice": "mira", "language": "English",
                                 "response_format": "pcm"})
            websocket.send_json({"type": "text", "text": "A quiet whisper."})
            websocket.send_json({"type": "end"})
            pcm = bytearray()
            for _ in range(40):
                event = websocket.receive()
                if event.get("bytes") is not None:
                    pcm.extend(event["bytes"])
                    continue
                payload = json.loads(event["text"])
                assert payload["type"] != "error", payload
                if payload["type"] == "done":
                    break
            else:
                pytest.fail("WebSocket did not finish")
            assert bytes(pcm) == engine.silent + engine.audible
        assert client.get("/health").json()["unusable_outputs_total"] == 0
