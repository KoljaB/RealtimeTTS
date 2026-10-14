"""Per-text languages are explicit streaming boundaries, not extra inference."""
import json

import pytest
from fastapi.testclient import TestClient

from RealtimeTTS.language_router import LanguageDetection, QwenLanguageRouter
from RealtimeTTS.qwen_server import create_app
from tests.test_qwen_server import FakeEngine, _persist_voice, _server


class RecordingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.calls = []
    def synthesize(self, text):
        self.calls.append((text, self.current_voice.language, self.current_voice.name))
        return super().synthesize(text)


class EnglishDetector:
    def __init__(self):
        self.calls = []
    def warmup(self, *_args):
        return LanguageDetection("english", "en", 1.0, 0.0)
    def detect(self, text):
        self.calls.append(text)
        assert "árbol" not in text, "Queued Spanish contaminated English lookahead"
        return LanguageDetection("english", "en", 1.0, 0.0)


def make_server(tmp_path):
    for voice in ("mira", "mira_es", "mira_de"):
        _persist_voice(tmp_path, name=voice)
    engine, detector = RecordingEngine(), EnglishDetector()
    router = QwenLanguageRouter(detector, voice_routes={"mira": {"en": "mira", "es": "mira_es", "de": "mira_de"}})
    return _server(tmp_path, engine, language_router=router), engine, detector


def drain(websocket):
    events = []
    while True:
        message = websocket.receive()
        if message.get("bytes") is not None:
            continue
        event = json.loads(message["text"])
        assert event["type"] != "error", event
        events.append(event)
        if event["type"] == "done":
            return events


@pytest.mark.parametrize("base", ["Auto", "English"])
@pytest.mark.parametrize("split_sentences", [False, True])
def test_english_spanish_english_and_voice_routes_in_one_stream(tmp_path, base, split_sentences):
    server, engine, detector = make_server(tmp_path)
    with TestClient(create_app(server)) as client:
        assert client.get("/v1/capabilities").json()["features"]["per_text_language"]
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({"type": "config", "voice": "mira", "language": base, "split_sentences": split_sentences})
            for text, language in [("Tree is ", base), ("el ", "es"), ("árbol", "Spanish"), (". Repeat it.", base)]:
                ws.send_json({"type": "text", "text": text, "language": language})
            ws.send_json({"type": "end"})
            drain(ws)
    assert engine.calls == [("Tree is", "english", "mira"), ("el árbol", "spanish", "mira_es"),
                            (". Repeat it.", "english", "mira")]
    assert len(detector.calls) == (1 if base == "Auto" else 0)


def test_explicit_first_span_does_not_select_language_for_later_auto(tmp_path):
    server, engine, detector = make_server(tmp_path)
    with TestClient(create_app(server)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({"type": "config", "voice": "mira", "language": "Auto", "split_sentences": False})
            ws.send_json({"type": "text", "text": "el árbol", "language": "es"})
            ws.send_json({"type": "text", "text": "Can you repeat it?", "language": "Auto"})
            ws.send_json({"type": "end"})
            drain(ws)
    assert engine.calls == [("el árbol", "spanish", "mira_es"), ("Can you repeat it?", "english", "mira")]
    assert detector.calls == ["Can you repeat it?"]


def test_language_boundary_flushes_before_end_or_next_sentence(tmp_path):
    server, engine, _ = make_server(tmp_path)
    with TestClient(create_app(server)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({"type": "config", "voice": "mira", "language": "English", "split_sentences": False})
            ws.send_json({"type": "text", "text": "Tree is "})
            ws.send_json({"type": "text", "text": "el árbol", "language": "es"})
            # No end, punctuation or flush: the explicit switch releases English.
            while True:
                message = ws.receive()
                if message.get("bytes"):
                    break
                assert json.loads(message["text"])["type"] != "error"
            assert engine.calls == [("Tree is", "english", "mira")]
            ws.send_json({"type": "end"})
            drain(ws)
    assert engine.calls[-1] == ("el árbol", "spanish", "mira_es")


@pytest.mark.parametrize("bad", [None, "", 7, True, "xx"])
def test_invalid_language_rejected_without_synthesis(tmp_path, bad):
    server, engine, _ = make_server(tmp_path)
    with TestClient(create_app(server)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({"type": "config", "voice": "mira", "language": "English"})
            ws.send_json({"type": "text", "text": "Never synthesize.", "language": bad})
            event = ws.receive_json()
            assert event["type"] == "error"
            assert event["error"]["type"] == "invalid_request_error"
    assert not engine.calls
