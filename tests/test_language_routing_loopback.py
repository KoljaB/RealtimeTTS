"""Exercise the actual HTTP/WebSocket adapter against local provider servers."""
from contextlib import contextmanager
import socket
import threading
import time

from fastapi.testclient import TestClient
import uvicorn

from RealtimeTTS import LanguageRoute, LanguageRoutingEngine, RemoteTTSEngine
from RealtimeTTS.language_routing_server import LanguageRoutingServer
from RealtimeTTS.qwen_server import QwenHttpServer, create_app
from tests.test_language_routing_server import Provider, drain


class ProviderServer(QwenHttpServer):
    def _validate_voice(self, value):
        assert value == "default"
        return value

    def list_voices(self):
        return [{"name": "default"}]

    def configure_engine(self, request):
        self.engine.set_voice(request.voice)
        self.engine.set_language(request.language)
        if self.engine.direct_voice_control:
            self.engine.set_instructions(request.instructions or "")

    def capabilities(self):
        value = super().capabilities()
        value["features"]["direct_voice_control"] = self.engine.direct_voice_control
        return value


@contextmanager
def serving(server):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    runner = uvicorn.Server(uvicorn.Config(
        create_app(server), host="127.0.0.1", port=port,
        log_level="error", lifespan="off", ws="wsproto",
    ))
    worker = threading.Thread(target=runner.run, kwargs={"sockets": [sock]}, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while not runner.started and worker.is_alive() and time.monotonic() < deadline:
            threading.Event().wait(.01)
        assert runner.started, "provider did not start"
        yield f"http://127.0.0.1:{port}"
    finally:
        runner.should_exit = True
        worker.join(5)
        sock.close()
        assert not worker.is_alive(), "provider did not stop"


def test_real_provider_connections_preserve_cues_voices_and_single_boundary_gaps(tmp_path):
    breeze, qwen = Provider("breeze", True), Provider("qwen")
    bserver = ProviderServer(breeze, voice_dir=tmp_path / "b")
    qserver = ProviderServer(qwen, voice_dir=tmp_path / "q")
    with serving(bserver) as b_url, serving(qserver) as q_url:
        b = RemoteTTSEngine(b_url, "default")
        q = RemoteTTSEngine(q_url, "default")
        router = LanguageRoutingEngine(
            {"en": [LanguageRoute(b, voice="default")],
             "*": [LanguageRoute(q, voice="default")]},
            pre_first_pcm_timeout=3,
        )
        app = LanguageRoutingServer(router, voice_dir=tmp_path / "r")
        with TestClient(create_app(app)) as client:
            with client.websocket_connect("/v1/audio/speech-stream") as ws:
                ws.send_json({"type": "config", "voice": "default", "split_sentences": False})
                ws.send_json({"type": "text", "text": "Tree is", "language": "en",
                              "instructions": "warm, encouraging"})
                ws.send_json({"type": "text", "text": "el árbol", "language": "es"})
                ws.send_json({"type": "text", "text": ". Try it.", "language": "en"})
                ws.send_json({"type": "end"})
                _, pcm = drain(ws)
        assert breeze.calls == [
            ("Tree is", "english", "default", "warm, encouraging"),
            ("Try it.", "english", "default", "warm, encouraging"),
        ]
        assert qwen.calls == [("el árbol", "spanish", "default", "")]
        assert [len(part) for part in pcm] == [160, 3840, 160, 14400, 160, 14400]
