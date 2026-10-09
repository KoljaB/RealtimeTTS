import hashlib
import json
import pytest
from fastapi.testclient import TestClient
from RealtimeTTS.breeze_diagnostics import write_failure
from RealtimeTTS.qwen_server import create_app
from tests.test_qwen_server import FakeEngine, _register, _server
from tests.test_breeze_engine_server import Socket, make_engine


@pytest.mark.parametrize("text,classification,has_wav", [
    ("empty", "no_pcm", False), ("silent", "all_zero", True),
    ("quiet", "below_audibility_threshold", True),
])
def test_unusable_audio_writes_reproduction_and_retains_quiet_pcm(tmp_path, text, classification, has_wav):
    class Engine(FakeEngine):
        def synthesize(self, text):
            if text == "quiet":
                self.queue.put(b"\x0a\x00" * 200)
                return True
            return super().synthesize(text)
        def synthesis_diagnostics(self):
            return {"blocks": [{"text": text, "instruction": "Whispering",
                                "parameters": {"seed": 44, "cfg_scale": 4}}]}
    server = _server(tmp_path, Engine())
    captured = []
    server.record_synthesis_failure = lambda request, state: captured.append(
        write_failure(tmp_path / "failures", request, state))
    with TestClient(create_app(server)) as client:
        _register(client)
        request = server.parse_speech({"input": text, "voice": "mira", "response_format": "pcm"})
        state = server.start_synthesis(request)
        assert state.done.wait(2)
    assert len(captured) == 1
    record = json.loads(captured[0].read_text(encoding="utf-8"))
    assert record["audio"]["classification"] == classification
    assert bool(record["audio"]["saved_wav"]) is has_wav
    assert record["request"]["input"] == text
    assert record["request_id"] == state.request_id
    assert record["engine"]["blocks"][0]["parameters"]["seed"] == 44
    assert record["audio"]["audibility_mean_abs_threshold"] == 80
    if text == "quiet":
        assert record["audio"]["mean_abs"] == 10
        assert record["audio"]["saved_bytes"] == 400


def test_native_diagnostics_keep_prefix_seed_directions_and_commands(monkeypatch):
    socket = Socket()
    engine, _ = make_engine(monkeypatch, socket)
    context = engine.create_stream_context()
    engine.set_stream_context(context)
    assert engine.synthesize("[Amused] The preceding sentence.")
    assert engine.synthesize("[Whispering] The quiet continuation.")
    snapshot = engine.synthesis_diagnostics()
    engine.close_stream_context(context)
    assert [block["text"] for block in snapshot["blocks"]] == [
        "The preceding sentence.", "The quiet continuation."]
    assert [block["session_reused"] for block in snapshot["blocks"]] == [False, True]
    assert snapshot["blocks"][1]["instruction"] == "Whispering"
    assert snapshot["blocks"][1]["parameters"]["seed"] == 44
    assert all(block["pcm_sha256"] == hashlib.sha256(b"\xe8\x03" * 200).hexdigest()
               for block in snapshot["blocks"])
    payloads = [command["payload"] for command in snapshot["commands"]]
    assert sum(payload["type"] == "start" for payload in payloads) == 1
    assert [payload["text"] for payload in payloads if payload["type"] == "text"] == [
        "The preceding sentence.", "The quiet continuation."]
