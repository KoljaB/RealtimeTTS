"""Pronunciation exchanges at the actual HTTP/WebSocket synthesis boundary."""
import json

import pytest
from fastapi.testclient import TestClient

from RealtimeTTS.language_router import LanguageDetection, QwenLanguageRouter
from RealtimeTTS.qwen_server import create_app
from RealtimeTTS.text_exchange import TextExchange
from tests.test_qwen_server import FakeEngine, _register, _server


RULES = {"en": {"miep": "meep", "Miep": "Meep"}}


@pytest.mark.parametrize("text,expected", [
    ("miep", "meep"), ("Miep, miep! Miep's Miep’s", "Meep, meep! Meep's Meep’s"),
    ("miep. (Miep) /miep/", "meep. (Meep) /meep/"),
    ("smiep mieps miep2 2miep miep_test _miep MIEP", "smiep mieps miep2 2miep miep_test _miep MIEP"),
    ("ämiep miepé 中miep miep中", "ämiep miepé 中miep miep中"),
    ("miep\u0301 \u0301miep", "miep\u0301 \u0301miep"),
])
def test_word_boundaries_and_case(text, expected):
    assert TextExchange(exchange_by_language=RULES).apply(text, "English")[0] == expected


def test_global_override_alias_longest_literal_and_non_cascading():
    rules = TextExchange({"Miep": "global", "API": "A P I", "A P I": "chained",
                          "C++": "see plus plus", "New": "old", "New York": "NY"},
                         {"en-US": {"Miep": "Meep"}})
    text = "Miep API C++ New York"
    assert rules.apply(text, "en")[0] == "Meep A P I see plus plus NY"
    assert rules.apply(text, "German")[0] == "global A P I see plus plus NY"


def test_auto_language_overrides_require_a_detector_but_explicit_language_does_not(tmp_path):
    engine = RecordingEngine()
    with TestClient(create_app(_server(tmp_path, engine))) as client:
        _register(client)
        payload = {"voice": "mira", "input": "Miep!", "exchange_by_language": RULES}
        result = client.post("/v1/audio/speech", json=payload)
        assert result.status_code == 400 and not engine.calls
        assert "language router" in result.text
        result = client.post("/v1/audio/speech", json={**payload, "language": "en-US"})
        assert result.status_code == 200
        assert engine.calls == [("Meep!", "english")]


def test_exact_source_ranges_survive_length_changes_and_unicode():
    text, ranges = TextExchange({"Miep": "Meep", "API": "A P I"}).apply("😀 Miep API!", "en")
    assert text == "😀 Meep A P I!"
    assert ranges[:7] == [[0, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]
    assert ranges[7:] == [[8, 9], [8, 10], [9, 10], [9, 11], [10, 11], [11, 12]]


class RecordingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.streaming_aligner = object()

    def synthesize(self, text):
        self.calls.append((text, self.current_voice.language))
        result = super().synthesize(text)
        callback = getattr(self, "on_alignment", None)
        if callback:
            callback({"type": "alignment_done"})
        return result


class Detector:
    def __init__(self, language):
        self.language, self.inputs = language, []

    def detect(self, text, *, fallback_language="english"):
        self.inputs.append(text)
        return LanguageDetection(self.language, self.language, 1.0, 0.0, False)


@pytest.mark.parametrize("language,expected", [("english", "Hello, Meep!"),
                                             ("german", "Hello, Miep!")])
def test_http_auto_detects_original_then_exchanges_only_selected_language(tmp_path, language, expected):
    detector, engine = Detector(language), RecordingEngine()
    server = _server(tmp_path, engine, language_router=QwenLanguageRouter(detector))
    with TestClient(create_app(server)) as client:
        _register(client)
        result = client.post("/v1/audio/speech", json={
            "voice": "mira", "input": "Hello, Miep!", "language": "Auto",
            "exchange_by_language": RULES,
        })
        assert result.status_code == 200 and result.content
    assert engine.calls == [(expected, language)]
    assert detector.inputs and all("Meep" not in text for text in detector.inputs)


def test_websocket_split_name_preserves_source_metadata(tmp_path):
    engine, detector = RecordingEngine(), Detector("english")
    server = _server(tmp_path, engine, language_router=QwenLanguageRouter(detector))
    with TestClient(create_app(server)) as client:
        _register(client)
        assert client.get("/v1/capabilities").json()["features"]["text_exchange"]
        with client.websocket_connect("/v1/audio/speech-stream") as websocket:
            websocket.send_json({"type": "config", "voice": "mira", "language": "Auto",
                                 "exchange_by_language": RULES})
            for text in ("Hello, Mi", "ep and mie", "p!"):
                websocket.send_json({"type": "text", "text": text})
            websocket.send_json({"type": "end"})
            events, audio = [], []
            while True:
                message = websocket.receive()
                if message.get("bytes"):
                    audio.append(message["bytes"])
                elif message.get("text"):
                    event = json.loads(message["text"])
                    events.append(event)
                    assert event["type"] != "error", event
                    if event["type"] == "done":
                        break
    assert audio and engine.calls
    assert " ".join(text for text, _ in engine.calls) == "Hello, Meep and meep!"
    starts = [event for event in events if event["type"] == "fragment_start"]
    assert starts and "".join(event["source_text"] for event in starts) == "Hello, Miep and miep!"
    assert all(len(event["source_ranges"]) == len(event["text"]) for event in starts)


@pytest.mark.parametrize("fields", [
    {"exchange": []}, {"exchange": {"": "x"}}, {"exchange": {"x": 4}},
    {"exchange": {"x": ""}}, {"exchange_by_language": []},
    {"exchange_by_language": {"auto": {"x": "y"}}},
    {"exchange_by_language": {"unknown": {"x": "y"}}},
    {"exchange_by_language": {"en": {}, "English": {}}},
])
def test_invalid_exchange_rejected_before_synthesis(tmp_path, fields):
    engine = RecordingEngine()
    with TestClient(create_app(_server(tmp_path, engine))) as client:
        _register(client)
        result = client.post("/v1/audio/speech", json={"voice": "mira", "input": "Hello", **fields})
        assert result.status_code == 400
        assert not engine.calls
