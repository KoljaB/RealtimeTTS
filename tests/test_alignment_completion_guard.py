"""Optional timestamp work must never hold completed speech open indefinitely."""

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from RealtimeTTS import qwen_server as api
from tests.test_qwen_server import FakeEngine, _register, _server


class IncompleteAlignmentEngine(FakeEngine):
    streaming_aligner = object()
    on_alignment = None
    last_alignment = None

    def __init__(self):
        super().__init__()
        self.missing_completion = True
        self.cancelled_alignments = []
        self.callbacks = []

    def synthesize(self, text):
        self.last_alignment = SimpleNamespace(
            finish=lambda *, cancelled=False: self.cancelled_alignments.append(cancelled)
        )
        self.callbacks.append(self.on_alignment)
        result = super().synthesize(text)
        self.on_alignment({
            "type": "alignment", "words": [{"word": text, "start_time": 0, "end_time": 0.1}],
            "characters": [], "complete": False,
        })
        if not self.missing_completion:
            self.on_alignment({"type": "alignment_done", "complete": True})
        return result


def _collect(client):
    events, pcm = [], []
    with client.websocket_connect("/v1/audio/speech-stream") as websocket:
        websocket.send_json({"type": "config", "voice": "mira", "language": "English"})
        websocket.send_json({"type": "text", "text": "The audio remains intact."})
        websocket.send_json({"type": "end"})
        while True:
            message = websocket.receive()
            if message.get("bytes") is not None:
                pcm.append(message["bytes"])
                continue
            event = json.loads(message["text"])
            events.append(event)
            assert event["type"] != "error", event
            if event["type"] == "done":
                return events, b"".join(pcm)


def test_missing_alignment_completion_preserves_audio_and_next_request(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ALIGNMENT_FINALIZE_TIMEOUT_SECONDS", 0.05)
    engine = IncompleteAlignmentEngine()
    server = _server(tmp_path, engine)
    with TestClient(api.create_app(server)) as client:
        _register(client)
        started = time.monotonic()
        events, pcm = _collect(client)
        assert time.monotonic() - started < 2.0
        assert pcm.startswith(engine.silent + engine.audible)
        assert any(event["type"] == "alignment" for event in events)
        completion = [event for event in events if event["type"] == "alignment_done"]
        assert len(completion) == 1
        assert completion[0]["complete"] is False
        assert completion[0]["finish_reason"] == "alignment_timeout"
        assert events[-1]["type"] == "done"
        assert engine.cancelled_alignments == [True]
        assert not engine.stopped

        # A late callback from the abandoned metadata job cannot affect its successor.
        engine.callbacks[0]({"type": "alignment_done", "complete": True})
        engine.missing_completion = False
        next_events, next_pcm = _collect(client)
        assert next_pcm == pcm
        assert [e["complete"] for e in next_events if e["type"] == "alignment_done"] == [True]
        assert engine.cancelled_alignments == [True]


def _state():
    state = api.SynthesisState(
        "pcm", request_id="test", output_queue_chunks=4,
        deadline=time.monotonic() + 10,
    )
    state.alignment_done.clear()
    return state


def _sender(state, events, cancelled):
    async def send_json(event):
        events.append(event)
    return api._send_alignment_events(
        SimpleNamespace(send_json=send_json), state,
        session_id="test", fragment_index=1, cancelled=cancelled,
        send_lock=asyncio.Lock(),
    )


def test_alignment_budget_starts_after_audio_and_keeps_late_success(monkeypatch):
    monkeypatch.setattr(api, "ALIGNMENT_FINALIZE_TIMEOUT_SECONDS", 0.08)
    async def run():
        state, events, cancelled = _state(), [], asyncio.Event()
        task = asyncio.create_task(_sender(state, events, cancelled))
        await asyncio.sleep(0.12)
        assert not task.done()  # Ongoing synthesis must not consume the metadata tail budget.
        state.done.set()
        state.alignment_events.put({"type": "alignment", "words": [{"word": "late"}]})
        state.alignment_events.put({"type": "alignment_done", "complete": True})
        state.alignment_done.set()
        await asyncio.wait_for(task, 1)
        assert [event["type"] for event in events] == ["alignment", "alignment_done"]
        assert events[-1]["complete"] is True
    asyncio.run(run())


def test_pending_alignment_cancellation_does_not_wait_for_tail(monkeypatch):
    monkeypatch.setattr(api, "ALIGNMENT_FINALIZE_TIMEOUT_SECONDS", 30)
    async def run():
        state, events, cancelled = _state(), [], asyncio.Event()
        state.done.set()
        task = asyncio.create_task(_sender(state, events, cancelled))
        await asyncio.sleep(0)
        cancelled.set()
        await asyncio.wait_for(task, 0.5)
        assert events == []
    asyncio.run(run())


def test_nonterminal_metadata_cannot_extend_tail_budget(monkeypatch):
    monkeypatch.setattr(api, "ALIGNMENT_FINALIZE_TIMEOUT_SECONDS", 0.05)
    async def run():
        state, events, cancelled = _state(), [], asyncio.Event()
        state.done.set()
        async def keep_busy():
            while True:
                state.alignment_events.put({"type": "alignment", "words": []})
                await asyncio.sleep(0.001)
        producer = asyncio.create_task(keep_busy())
        try:
            await asyncio.wait_for(_sender(state, events, cancelled), 0.5)
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
        assert events[-1]["type"] == "alignment_done"
        assert events[-1]["complete"] is False
        assert events[-1]["finish_reason"] == "alignment_timeout"
    asyncio.run(run())
