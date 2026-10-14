
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from RealtimeTTS.language_router import FastTextLanguageDetector, LanguageDetection, QwenLanguageRouter
from RealtimeTTS.mixed_language import split_mixed_language
from RealtimeTTS.qwen_server import ApiError, create_app, _stream_mixed_fragments
from tests.test_qwen_language_spans import RecordingEngine, drain
from tests.test_qwen_server import _persist_voice, _server

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "fasttext_tutor_boundaries.json").read_text(encoding="utf8"))


class RecordedDetector(FastTextLanguageDetector):
    """Real lid.176.bin scores, without a model download in the test suite."""
    def __init__(self, row):
        self.row = row
        self.fallback_language = "english"
    def warmup(self, *_args):
        return self.detect(self.row["text"])
    def detect(self, text):
        record = self.row["records"].get(text, {})
        if "detection" in record:
            return LanguageDetection(**record["detection"])
        return LanguageDetection("english", "en", 1.0, 0.0,
                                 candidate_language="english", confidence_margin=1.0)
    def language_probabilities(self, text):
        return self.row["records"][text]["scores"]


@pytest.mark.parametrize("row", FIXTURE["rows"], ids=lambda row: row["text"])
def test_recorded_real_fasttext_boundaries_preserve_every_character(row):
    spans = split_mixed_language(row["text"], row["target"], RecordedDetector(row))
    assert "".join(span.text for span in spans) == row["text"]
    assert [span.text for span in spans if span.language] == row["expected"]


@pytest.mark.parametrize("row", FIXTURE["rows"][:6] + FIXTURE["rows"][8:], ids=lambda row: row["text"])
def test_characterwise_websocket_recovers_missing_controls(tmp_path, row):
    detector = RecordedDetector(row)
    engine = RecordingEngine()
    _persist_voice(tmp_path)
    server = _server(tmp_path, engine, language_router=QwenLanguageRouter(detector))
    with TestClient(create_app(server)) as client:
        assert client.get("/v1/capabilities").json()["features"]["mixed_language_fallback"]
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({"type":"config", "voice":"mira", "language":"Auto",
                          "mixed_language":row["target"], "split_sentences":False})
            for char in row["text"]:
                ws.send_json({"type":"text", "text":char})
            ws.send_json({"type":"end"})
            drain(ws)
    assert [text.rstrip(".!?") for text, language, _ in engine.calls if language == row["target"]] == row["expected"]
    assert all(language in ("english", "german", row["target"]) for _, language, _ in engine.calls)


def test_explicit_language_and_disabled_mode_do_not_use_detector():
    class NeverDetector:
        def detect(self, *_args):
            raise AssertionError("Explicit or disabled routing must bypass detection")
    async def run(language, target):
        async def fragments():
            yield "as flores"
        return [item async for item in _stream_mixed_fragments(
            fragments(), language=language, target=target, detector=NeverDetector())]
    assert asyncio.run(run("portuguese", "portuguese")) == ["as flores"]
    assert asyncio.run(run("auto", None)) == ["as flores"]


def test_repair_never_requests_next_fragment_before_releasing_current():
    row = FIXTURE["rows"][0]
    async def run():
        async def source():
            yield row["text"]
            raise AssertionError("Repair must not look ahead")
        stream = _stream_mixed_fragments(source(), language="auto",
                                        target=row["target"], detector=RecordedDetector(row))
        first = await anext(stream)
        await stream.aclose()
        return first
    assert asyncio.run(run()) == "Leash is "


@pytest.mark.parametrize("text", ["", "x " * 130, "a" * 2049])
def test_bounded_input_does_not_classify(text):
    assert split_mixed_language(text, "spanish", None)[0].text == text


def test_fasttext_probability_input_flattens_newlines():
    calls = []
    class Model:
        def predict(self, text, k):
            calls.append((text, k))
            return ["__label__es", "__label__en"], [.9, .1]
    detector = FastTextLanguageDetector(model=Model())
    assert detector.language_probabilities("el\nárbol") == {"spanish":.9, "english":.1}
    assert calls == [("el árbol", -1)]


@pytest.mark.parametrize("language,target", [("English","es"), ("Auto","fr"), ("Auto",True)])
def test_invalid_mixed_configuration_is_rejected(tmp_path, language, target):
    row = FIXTURE["rows"][0]
    _persist_voice(tmp_path)
    server = _server(tmp_path, RecordingEngine(), language_router=QwenLanguageRouter(RecordedDetector(row)))
    with pytest.raises(ApiError, match="Auto|Spanish|language"):
        server.parse_speech_options({"voice":"mira", "language":language, "mixed_language":target})


@pytest.mark.parametrize("opening,closing", [('"','"'), ("“","”"), ("„","“"), ("'","'")])
def test_quotes_bound_language_island_and_prefer_complete_phrase(opening, closing):
    class Scores:
        fallback_language = "english"
        def detect(self, text):
            return LanguageDetection("english", "en", 1.0, 0.0, candidate_language="english")
        def language_probabilities(self, text):
            assert opening not in text and closing not in text, "Crossed a quotation boundary"
            return {
                "buenos días":{"spanish":.85,"english":.01},
                "días":{"spanish":.999,"english":.0001},
            }.get(text, {"english":.99})
    text = "Try " + opening + "buenos días" + closing + " once."
    spans = split_mixed_language(text, "spanish", Scores())
    assert [span.text for span in spans if span.language] == ["buenos días"]
    assert "".join(span.text for span in spans) == text


def test_recovered_phrase_keeps_terminal_punctuation_for_synthesis():
    row = FIXTURE["rows"][3]
    async def run():
        async def fragments():
            yield row["text"]
        return [item async for item in _stream_mixed_fragments(
            fragments(), language="auto", target=row["target"], detector=RecordedDetector(row))]
    spans = asyncio.run(run())
    assert "".join(spans) == row["text"]
    assert spans[-1] == "Veo un árbol."
    assert spans[-1].language == "spanish"
