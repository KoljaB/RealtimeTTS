"""Ordered, cancellable narration segments on one synthesis session."""
import json
import threading

import pytest
from fastapi.testclient import TestClient

from tests.test_qwen_server import FakeEngine, _server, _register, create_app


def config(ws):
    ws.send_json({"type": "config", "voice": "mira", "language": "en", "response_format": "pcm"})


def segment(ws, number, text=""):
    ws.send_json({"type": "segment_start", "segment_id": number})
    if text:
        ws.send_json({"type": "text", "text": text})
    ws.send_json({"type": "flush", "segment_id": number})


def until(ws, wanted):
    events = []
    while True:
        message = ws.receive()
        assert message["type"] != "websocket.close", message
        event = {"type": "pcm", "data": message["bytes"]} if message.get("bytes") is not None else json.loads(message["text"])
        events.append(event)
        assert event["type"] != "error", event
        if event["type"] == wanted:
            return events


class RecordingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        return super().synthesize(text)


def test_two_segments_and_empty_segment_are_ordered(tmp_path):
    engine = RecordingEngine()
    with TestClient(create_app(_server(tmp_path, engine))) as client:
        _register(client)
        assert client.get("/v1/capabilities").json()["stream_controls"]["segmented_narration"] is True
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            config(ws)
            segment(ws, 1, "First spoken update.")
            segment(ws, 2)
            segment(ws, 3, "The final answer.")
            ws.send_json({"type": "end"})
            events = until(ws, "done")
    boundaries = [e for e in events if e["type"].startswith("segment_")]
    assert boundaries == [event for number in (1, 2, 3) for event in (
        {"type": "segment_started", "segment_id": number},
        {"type": "segment_done", "segment_id": number, "skipped": False},
    )]
    active = None
    for event in events:
        if event["type"] == "segment_started":
            active = event["segment_id"]
        elif event["type"] == "segment_done":
            active = None
        elif event["type"] == "pcm":
            assert active in (1, 3), "PCM escaped its segment boundary"
    assert engine.calls == ["First spoken update.", "The final answer."]


def test_skip_active_and_queued_remainder_then_stale_skip_preserves_successor(tmp_path):
    class ControlledEngine(RecordingEngine):
        def __init__(self):
            super().__init__()
            self.first_started = threading.Event()
            self.first_stopped = threading.Event()
            self.successor_started = threading.Event()
            self.successor_release = threading.Event()
            self.stop_calls = 0
            self.first_finished = False

        def synthesize(self, text):
            if not self.calls:
                self.calls.append(text)
                self.queue.put(self.audible)
                self.first_started.set()
                assert self.first_stopped.wait(3)
                self.queue.put(self.audible)  # Late native PCM must be discarded.
                self.first_finished = True
                return True
            assert self.first_finished, "successor overlapped cancelled worker"
            self.calls.append(text)
            self.successor_started.set()
            assert self.successor_release.wait(3)
            return FakeEngine.synthesize(self, text)

        def stop(self):
            self.stop_calls += 1
            self.first_stopped.set()

    engine = ControlledEngine()
    with TestClient(create_app(_server(tmp_path, engine))) as client:
        _register(client)
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            config(ws)
            ws.send_json({"type": "segment_start", "segment_id": 1})
            ws.send_json({"type": "text", "text": "First spoken update."})
            ws.send_json({"type": "flush"})
            assert engine.first_started.wait(2)
            ws.send_json({"type": "text", "text": "This queued remainder must never be spoken."})
            ws.send_json({"type": "flush", "segment_id": 1})
            segment(ws, 2, "This entire pending segment must be skipped.")
            ws.send_json({"type": "skip_segment", "segment_id": 2})
            segment(ws, 3, "The final answer.")
            ws.send_json({"type": "skip_segment", "segment_id": 1})
            first = until(ws, "segment_done")
            assert first[-1] == {"type": "segment_done", "segment_id": 1, "skipped": True}
            queued = until(ws, "segment_done")
            assert queued == [{"type": "segment_started", "segment_id": 2}, {"type": "segment_done", "segment_id": 2, "skipped": True}]
            assert engine.successor_started.wait(2)
            ws.send_json({"type": "skip_segment", "segment_id": 1})
            # A subsequent acknowledged pause confirms the stale skip was processed.
            ws.send_json({"type": "resume"})
            until(ws, "resumed")
            assert engine.stop_calls == 1
            engine.successor_release.set()
            ws.send_json({"type": "end"})
            final = until(ws, "done")
            assert {"type": "segment_done", "segment_id": 3, "skipped": False} in final
            assert any(event["type"] == "pcm" for event in final)
    assert engine.calls == ["First spoken update.", "The final answer."]


@pytest.mark.parametrize("number", [0, -1, True, "1", None])
def test_segment_ids_must_be_positive_integers(tmp_path, number):
    with TestClient(create_app(_server(tmp_path))) as client:
        _register(client)
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            config(ws)
            ws.send_json({"type": "segment_start", "segment_id": number})
            event = json.loads(ws.receive()["text"])
            assert event["type"] == "error"
