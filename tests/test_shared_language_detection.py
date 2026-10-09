"""Regression coverage for classification shared by Qwen and Breeze."""
import threading

import pytest
from fastapi.testclient import TestClient

from RealtimeTTS.breeze_server import BreezeHttpServer, build_argument_parser
from RealtimeTTS.language_router import FastTextLanguageDetector, LanguageDetection
from RealtimeTTS.language_router import QwenLanguageRouter
from RealtimeTTS.qwen_server import create_app
from tests.test_language_router import FakeFastTextModel
from tests.test_qwen_server import _server


class Detector:
    def __init__(self):
        self.calls = []
        self.warmups = 0

    def warmup(self, text="warmup"):
        self.warmups += 1

    def detect(self, text):
        self.calls.append(text)
        return LanguageDetection("german", "de", .99, .01)


def breeze_server(tmp_path, detector=None):
    engine = _server(tmp_path / "engine").engine
    engine.cfg_scale = 4.0
    return BreezeHttpServer(engine, voice_dir=tmp_path / "voices",
                            api_key="test-key", language_detector=detector)


@pytest.mark.parametrize("backend", ["qwen", "breeze"])
def test_both_servers_classify_without_synthesis_routing(tmp_path, backend):
    detector = Detector()
    server = (breeze_server(tmp_path, detector) if backend == "breeze"
              else _server(tmp_path, api_key="test-key", language_detector=detector))
    assert server.language_router is None
    with TestClient(create_app(server)) as client:
        assert detector.warmups == 1
        headers = {"Authorization": "Bearer test-key"}
        assert client.post("/v1/text/languages", json={"texts": ["Guten Tag."]}).status_code == 401
        response = client.post("/v1/text/languages",
                               json={"texts": ["Guten Tag.", "Wie geht es dir?"]}, headers=headers)
        assert response.status_code == 200, response.text
        assert [x["language"] for x in response.json()["languages"]] == ["german", "german"]
        assert detector.calls == ["Guten Tag.", "Wie geht es dir?"]
        features = client.get("/v1/capabilities", headers=headers).json()["features"]
        assert "german" in response.json()["supported_languages"]
        if backend == "breeze":
            assert features["supported_languages"] == ["en", "zh"]
    assert detector.warmups == 1


def test_qwen_router_and_endpoint_share_one_warmed_detector(tmp_path):
    detector = Detector()
    server = _server(tmp_path, language_router=QwenLanguageRouter(detector))
    with TestClient(create_app(server)) as client:
        assert server.language_detector is detector
        assert detector.warmups == 1
        response = client.post("/v1/text/languages", json={"texts": ["Guten Tag."]})
        assert response.status_code == 200
        route = server.language_router.resolve("Guten Tag.", base_voice="test")
        assert route.language == "german"
        assert detector.calls == ["Guten Tag.", "Guten Tag."]
        server.startup()
        assert detector.warmups == 1


def test_classification_does_not_wait_for_synthesis_lock(tmp_path):
    detector = Detector()
    server = breeze_server(tmp_path, detector)
    acquired, release = threading.Event(), threading.Event()

    def hold_engine():
        with server._operation_lock:
            acquired.set()
            release.wait(5)

    with TestClient(create_app(server)) as client:
        worker = threading.Thread(target=hold_engine)
        worker.start()
        assert acquired.wait(1)
        try:
            response = client.post("/v1/text/languages", json={"texts": ["Guten Tag."]},
                                   headers={"Authorization": "Bearer test-key"})
            assert response.status_code == 200
            assert worker.is_alive()
        finally:
            release.set()
            worker.join(2)


def test_breeze_fallback_stays_uncertain_and_missing_detector_is_advertised(tmp_path):
    detector = FastTextLanguageDetector(model=FakeFastTextModel(
        ("__label__de", "__label__en"), (.55, .40)))
    server = breeze_server(tmp_path, detector)
    with TestClient(create_app(server)) as client:
        response = client.post("/v1/text/languages",
            json={"texts": ["Ja."], "fallback_language": "french"},
            headers={"Authorization": "Bearer test-key"})
        assert response.status_code == 200
        result = response.json()["languages"][0]
        assert result["language"] == "french"
        assert result["used_fallback"] is True
    missing = breeze_server(tmp_path / "missing")
    assert missing.language_detector is None


def test_breeze_accepts_explicit_model_path_and_failed_warmup_prevents_ready(tmp_path):
    args = build_argument_parser().parse_args(["--language-id-model", str(tmp_path / "lid.176.bin")])
    assert args.language_id_model == tmp_path / "lid.176.bin"
    detector = FastTextLanguageDetector(args.language_id_model,
                                       model_loader=lambda path: (_ for _ in ()).throw(
                                           FileNotFoundError(path)))
    server = breeze_server(tmp_path, detector)
    with pytest.raises(FileNotFoundError):
        server.startup()
    assert server.is_ready() is False
