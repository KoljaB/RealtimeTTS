from fastapi.testclient import TestClient
from RealtimeTTS.language_router import LanguageDetection, QwenLanguageRouter
from RealtimeTTS.qwen_server import create_app
from tests.test_qwen_server import _server


class Detector:
    def __init__(self):
        self.texts = []
    def detect(self, text):
        self.texts.append(text)
        return LanguageDetection("german", "__label__de", .99, .01)


def test_language_endpoint_reuses_detector_and_requires_authentication(tmp_path):
    detector = Detector()
    server = _server(tmp_path, api_key="test-key",
                     language_router=QwenLanguageRouter(detector))
    with TestClient(create_app(server)) as client:
        detector.texts.clear()
        assert client.post("/v1/text/languages", json={"texts": ["Hallo."]}).status_code == 401
        response = client.post("/v1/text/languages", json={"texts": ["Hallo.", "Guten Tag."]},
                               headers={"Authorization": "Bearer test-key"})
        assert response.status_code == 200, response.text
        data = response.json()
        assert [x["language"] for x in data["languages"]] == ["german", "german"]
        assert detector.texts == ["Hallo.", "Guten Tag."]
        assert len(data["supported_languages"]) == 10


def test_language_endpoint_rejects_invalid_batches_and_missing_detector(tmp_path):
    with TestClient(create_app(_server(tmp_path))) as client:
        assert client.post("/v1/text/languages", json={"texts": []}).status_code == 400
        assert client.post("/v1/text/languages", json={"texts": [""]}).status_code == 400
        assert client.post("/v1/text/languages", json={"texts": ["x"] * 129}).status_code == 400
        assert client.post("/v1/text/languages", json={"texts": ["Hello."]}).status_code == 503
