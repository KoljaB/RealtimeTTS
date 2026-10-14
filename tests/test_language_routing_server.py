"""Protocol integration for English Breeze and foreign-language Qwen routes."""
import json

import pytest
from fastapi.testclient import TestClient

from RealtimeTTS import LanguageRoute, LanguageRoutingEngine
from RealtimeTTS.engines.base_engine import BaseEngine
from RealtimeTTS.language_routing_server import LanguageRoutingServer
from RealtimeTTS.qwen_server import create_app


class Provider(BaseEngine):
    def __init__(self, name, instructions=False):
        self.name = name
        self.direct_voice_control = self.instruction_control = instructions
        self.calls = []
        self.voice = None
        self.language = "en"
        self.instructions = ""
        self.fail = False
        self.closed = False

    def post_init(self):
        self.engine_name = self.name

    def get_stream_info(self):
        return 8, 1, 24000

    def set_voice(self, voice):
        self.voice = voice

    def set_language(self, language):
        self.language = language

    def set_instructions(self, instructions):
        assert self.direct_voice_control
        self.instructions = instructions

    def synthesize(self, text, sentence_count=0):
        self.calls.append((text, self.language, self.voice, self.instructions))
        if self.fail:
            return False
        self.queue.put(b"\x20\x03" * 80)
        return True

    def shutdown(self):
        self.closed = True


def server(tmp_path):
    breeze, qwen = Provider("breeze", True), Provider("qwen")
    b = LanguageRoute(breeze, voice="b-default", voices={"mira": "b-mira"})
    q = LanguageRoute(qwen, voice="q-default", voices={"mira": "q-mira"})
    router = LanguageRoutingEngine({"en": [b, q], "zh": [b, q], "*": [q]})
    return LanguageRoutingServer(router, providers={"breeze": breeze, "qwen": qwen},
                                 voice_dir=tmp_path), breeze, qwen


def drain(ws):
    events, pcm = [], []
    while True:
        message = ws.receive()
        if message.get("bytes") is not None:
            pcm.append(message["bytes"])
        else:
            event = json.loads(message["text"])
            assert event["type"] != "error", event
            events.append(event)
            if event["type"] == "done":
                return events, pcm


def test_languages_keep_delivery_and_provider_voice_isolation(tmp_path):
    app, breeze, qwen = server(tmp_path)
    with TestClient(create_app(app)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({"type": "config", "voice": "mira", "language": "Auto", "split_sentences": False})
            for text, language in [("Tree is", "en"), ("el árbol", "es"), (". Try it.", "en")]:
                ws.send_json({"type": "text", "text": text, "language": language,
                              "instructions": "warm, encouraging"})
            ws.send_json({"type": "end"})
            _, pcm = drain(ws)
        # A new reply without instructions resets the old cue.
        response = client.post("/v1/audio/speech", json={"voice": "mira", "input": "Again.", "language": "en"})
        assert response.status_code == 200
    assert breeze.calls == [("Tree is", "en", "b-mira", "warm, encouraging"),
                            ("Try it.", "en", "b-mira", "warm, encouraging"),
                            ("Again.", "en", "b-mira", "")]
    assert qwen.calls == [("el árbol", "es", "q-mira", "")]
    assert [len(part) for part in pcm] == [160, 3840, 160, 14400, 160, 14400]
    assert not breeze.closed and not qwen.closed


def test_failed_english_primary_uses_finite_configured_fallback(tmp_path):
    app, breeze, qwen = server(tmp_path)
    breeze.fail = True
    with TestClient(create_app(app)) as client:
        response = client.post("/v1/audio/speech", json={
            "voice": "mira", "input": "Hello.", "language": "English", "instructions": "warm",
        })
        assert response.status_code == 200
    assert len(breeze.calls) == len(qwen.calls) == 1
    assert qwen.calls[0] == ("Hello.", "en", "q-mira", "")


def test_capabilities_and_voices_describe_routes_without_native_defaults(tmp_path):
    app, _, _ = server(tmp_path)
    with TestClient(create_app(app)) as client:
        capabilities = client.get("/v1/capabilities").json()
        assert capabilities["engine"]["name"] == "language-routing"
        assert capabilities["features"]["language_engine_routing"]
        assert capabilities["features"]["per_text_language"]
        assert not capabilities["features"]["streaming_alignment"]
        assert capabilities["routing"]["routes"]["en"] == ["breeze", "qwen"]
        assert capabilities["sampling_defaults"] == {}
        voices = client.get("/v1/audio/voices").json()["voices"]
        assert next(v for v in voices if v["name"] == "mira")["providers"] == {
            "breeze": "b-mira", "qwen": "q-mira",
        }


@pytest.mark.parametrize("field,value", [("seed", 42), ("temperature", .7), ("max_new_tokens", 100)])
def test_sampling_overrides_do_not_silently_change_provider_defaults(tmp_path, field, value):
    app, breeze, qwen = server(tmp_path)
    with TestClient(create_app(app)) as client:
        response = client.post("/v1/audio/speech", json={
            "voice": "mira", "input": "Hello.", field: value,
        })
        assert response.status_code == 400
    assert not breeze.calls and not qwen.calls
