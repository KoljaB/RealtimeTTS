"""An em dash releases a natural opening phrase before the answer finishes."""
import threading

import pytest
from fastapi.testclient import TestClient

from RealtimeTTS import TextToAudioStream
from tests.test_qwen_segmented_narration import config, until
from tests.test_qwen_server import FakeEngine, _register, _server, create_app
from tests.test_text_to_stream_alignment import _FakePyAudio, _PCMEngine
import RealtimeTTS.stream_player as stream_player


@pytest.mark.parametrize("prefix", ["Testing away then —", "Testing away then—"])
def test_qwen_em_dash_starts_synthesis_before_remainder_or_end(tmp_path, prefix):
    class RecordingEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.started = threading.Event()

        def synthesize(self, text):
            self.calls.append(text)
            self.started.set()
            return super().synthesize(text)

    engine = RecordingEngine()
    with TestClient(create_app(_server(tmp_path, engine))) as client:
        _register(client)
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            config(ws)
            for chunk in (prefix[:-1], prefix[-1], " "):
                ws.send_json({"type": "text", "text": chunk})
            assert engine.started.wait(3), "TTS waited beyond the em dash"
            assert engine.calls == [prefix]
            ws.send_json({"type": "text", "text": "I'm here and ready to go."})
            ws.send_json({"type": "end"})
            until(ws, "done")
    assert engine.calls == [prefix, "I'm here and ready to go."]


def test_text_to_audio_stream_default_splits_em_dash(monkeypatch):
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)
    fragments = []
    stream = TextToAudioStream(_PCMEngine(), tokenizer="rule-based")
    stream.feed("Testing away then — I'm here and ready to go.").play(
        before_sentence_synthesized=fragments.append,
    )
    assert fragments == ["Testing away then —", "I'm here and ready to go."]
