"""WebSocket regression for punctuation outside a language tag."""
import asyncio
import json
import queue
import threading

import pytest
from fastapi.testclient import TestClient

import RealtimeTTS.qwen_server as api
from RealtimeTTS._audio_formats import INT16
from RealtimeTTS.engines.base_engine import TimingInfo
from RealtimeTTS.qwen_server import create_app
from tests.test_qwen_language_spans import make_server
from tests.test_qwen_server import FakeEngine, _persist_voice, _server


class NativeAlignmentEngine(FakeEngine):
    provides_word_timings = True

    def __init__(self):
        super().__init__()
        self.timings = queue.Queue()

    def get_stream_info(self):
        return INT16, 1, 24_000

    def synthesize(self, text):
        result = super().synthesize(text)
        self.timings.put(TimingInfo(0.0, 0.1, text))
        return result


@pytest.mark.parametrize("suffix,gap", [(". Try it.", .30), (", try it.", .15), (" Try it.", .08)])
def test_gap_precedes_returning_english_without_resynthesizing_punctuation(tmp_path, suffix, gap):
    server, engine, _ = make_server(tmp_path)
    pcm = []
    with TestClient(create_app(server)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({"type": "config", "voice": "mira", "language": "en", "split_sentences": False})
            ws.send_json({"type": "text", "text": "el árbol", "language": "es"})
            ws.send_json({"type": "text", "text": suffix, "language": "en"})
            ws.send_json({"type": "end"})
            while True:
                message = ws.receive()
                if message.get("bytes") is not None:
                    pcm.append(message["bytes"])
                else:
                    event = json.loads(message["text"])
                    assert event["type"] != "error", event
                    if event["type"] == "done":
                        break
    assert engine.calls[0][0] == "el árbol"
    assert engine.calls[1][0] == suffix.lstrip("., ")
    assert pcm[2] == bytes(round(24000 * gap) * 2)


def test_skip_during_language_gap_does_not_start_cancelled_fragment(tmp_path, monkeypatch):
    server, engine, _ = make_server(tmp_path)
    entered_gap = threading.Event()
    positive_gap_calls = 0
    original_send_silence = api._send_stream_silence

    async def hold_second_gap(websocket, duration, *, cancelled, resumed, send_lock):
        nonlocal positive_gap_calls
        if duration > 0:
            positive_gap_calls += 1
            if positive_gap_calls == 1:
                entered_gap.set()
                while not cancelled.is_set():
                    await asyncio.sleep(0.001)
                return 0.0
        return await original_send_silence(
            websocket, duration, cancelled=cancelled, resumed=resumed, send_lock=send_lock
        )

    monkeypatch.setattr(api, "_send_stream_silence", hold_second_gap)
    with TestClient(create_app(server)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as websocket:
            websocket.send_json({"type": "config", "voice": "mira", "language": "en", "split_sentences": False})
            websocket.send_json({"type": "segment_start", "segment_id": 1})
            websocket.send_json({"type": "text", "text": "el árbol", "language": "es"})
            websocket.send_json({"type": "text", "text": "Try it.", "language": "en"})
            websocket.send_json({"type": "end"})
            assert entered_gap.wait(1.0)
            websocket.send_json({"type": "skip_segment", "segment_id": 1})
            while True:
                message = websocket.receive()
                if message.get("bytes") is not None:
                    continue
                event = json.loads(message["text"])
                assert event["type"] != "error", event
                if event["type"] == "done":
                    break
    assert engine.calls == [("el árbol", "spanish", "mira_es")]


def test_language_boundary_alignment_keeps_original_utf16_source_ranges(tmp_path):
    _persist_voice(tmp_path)
    engine = NativeAlignmentEngine()
    server = _server(tmp_path, engine)
    events = []
    with TestClient(create_app(server)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as websocket:
            websocket.send_json({
                "type": "config", "voice": "mira", "language": "en",
                "split_sentences": False,
                "alignment": {"source": "native", "level": "word"},
            })
            websocket.send_json({"type": "text", "text": "el árbol", "language": "es"})
            websocket.send_json({"type": "text", "text": ". 🙂 Try it.", "language": "en"})
            websocket.send_json({"type": "end"})
            while True:
                message = websocket.receive()
                if message.get("bytes") is not None:
                    continue
                event = json.loads(message["text"])
                assert event["type"] != "error", event
                events.append(event)
                if event["type"] == "done":
                    break
    fragments = [event for event in events if event["type"] == "fragment_start"]
    assert len(fragments) == 2
    returning = fragments[1]
    assert returning["text"] == ". 🙂 Try it."
    assert returning["spoken_text"] == "🙂 Try it."
    assert returning["source_text"] == ". 🙂 Try it."
    assert returning["source_ranges"][0] == [2, 4]


def test_separately_flushed_period_is_not_dropped_or_doubled(tmp_path):
    server, engine, _ = make_server(tmp_path)
    pcm = []
    with TestClient(create_app(server)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({"type": "config", "voice": "mira", "language": "en", "split_sentences": False})
            for text, language in [("el árbol", "es"), (".", "en"), ("Try it.", "en")]:
                ws.send_json({"type": "text", "text": text, "language": language})
                ws.send_json({"type": "flush"})
            ws.send_json({"type": "end"})
            while True:
                message = ws.receive()
                if message.get("bytes") is not None:
                    pcm.append(message["bytes"])
                else:
                    event = json.loads(message["text"])
                    assert event["type"] != "error", event
                    if event["type"] == "done":
                        break
    assert [call[0] for call in engine.calls] == ["el árbol", "Try it."]
    assert [len(chunk) for chunk in pcm] == [32, 128, 14400, 32, 128, 14400]
