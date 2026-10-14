"""Mock-provider contract tests for the persistent RemoteTTSEngine adapter."""
import json
import queue
import threading
import time

import pytest

import RealtimeTTS.engines.remote_tts_engine as remote
from RealtimeTTS.engines.remote_tts_engine import RemoteTTSEngine


PCM = b"\x01\x00" * 8


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeWire:
    def __init__(self, on_flush):
        self.on_flush = on_flush
        self.incoming = queue.Queue()
        self.sent = []
        self.flush_seen = threading.Event()
        self.closed = False

    def send(self, payload):
        event = json.loads(payload)
        self.sent.append(event)
        if event.get("type") == "flush":
            self.flush_seen.set()
            self.on_flush(self, event)

    def settimeout(self, _timeout):
        pass

    def recv(self):
        message = self.incoming.get()
        return "" if message is None else message

    def shutdown(self):
        self.incoming.put(None)

    def close(self, timeout=3):
        del timeout
        self.closed = True
        self.incoming.put(None)


def _capabilities(*, alignment=False, per_text_language=True, direct=False):
    return {
        "features": {
            "per_request_silence": True,
            "per_text_language": per_text_language,
            "direct_voice_control": direct,
            "streaming_alignment": alignment,
        },
        "stream_controls": {"segmented_narration": True},
        "audio": {"sample_rate_hz": 24_000, "channels": 1, "encoding": "s16le"},
    }


def _install(monkeypatch, factories, *, alignment=False, per_text_language=True, direct=False):
    created = []
    payload = _capabilities(
        alignment=alignment, per_text_language=per_text_language, direct=direct
    )
    monkeypatch.setattr(remote.requests, "get", lambda *args, **kwargs: Response(payload))

    def create_connection(*args, **kwargs):
        del args, kwargs
        wire = FakeWire(factories.pop(0))
        created.append(wire)
        return wire

    monkeypatch.setattr(remote.websocket, "create_connection", create_connection)
    return created


def _success(wire, flush):
    wire.incoming.put(PCM)
    wire.incoming.put(json.dumps({"type": "segment_done", "segment_id": flush["segment_id"]}))


def _engine(monkeypatch, factories, **kwargs):
    created = _install(monkeypatch, factories, **kwargs)
    engine = RemoteTTSEngine("http://provider.test", "mira", language="en")
    context = engine.create_stream_context()
    engine.set_stream_context(context)
    return engine, context, created


def test_persistent_connection_uses_segments_and_provider_silence_is_disabled(monkeypatch):
    engine, context, created = _engine(monkeypatch, [_success], per_text_language=True)
    try:
        assert engine.synthesize("one")
        assert engine.queue.get(timeout=1) == PCM
        engine.set_language("de")
        assert engine.synthesize("zwei")
        assert engine.queue.get(timeout=1) == PCM
        assert len(created) == 1
        sent = created[0].sent
        assert [event["type"] for event in sent] == [
            "config", "segment_start", "text", "flush",
            "segment_start", "text", "flush",
        ]
        assert sent[0]["silence"] == {"comma": 0, "sentence": 0, "transition": 0}
        assert [event["segment_id"] for event in sent if event["type"] == "segment_start"] == [1, 2]
        assert [event["segment_id"] for event in sent if event["type"] == "flush"] == [1, 2]
        text_events = [event for event in sent if event["type"] == "text"]
        assert [event["language"] for event in text_events] == ["english", "german"]
    finally:
        engine.close_stream_context(context)


def test_native_alignment_can_arrive_after_pcm_and_segment_completion(monkeypatch):
    def delayed_alignment(wire, flush):
        wire.incoming.put(json.dumps({"type": "fragment_start", "request_id": "req-1"}))
        wire.incoming.put(PCM)
        wire.incoming.put(json.dumps({"type": "segment_done", "segment_id": flush["segment_id"]}))

    engine, context, created = _engine(monkeypatch, [delayed_alignment], alignment=True)
    events = []
    engine.on_alignment = events.append
    try:
        assert engine.synthesize("hello")
        assert events == []
        created[0].incoming.put(json.dumps({
            "type": "alignment", "request_id": "req-1",
            "words": [{"word": "hello", "start_time": 0.0, "end_time": 0.1}],
        }))
        created[0].incoming.put(json.dumps({
            "type": "alignment_done", "request_id": "req-1", "complete": True,
        }))
        deadline = time.monotonic() + 1
        while len(events) < 2:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert [event["type"] for event in events] == ["alignment", "alignment_done"]
        assert all(event["engine_audio_offset"] == 0 for event in events)
    finally:
        engine.close_stream_context(context)


@pytest.mark.parametrize("phase", ["before_pcm", "after_pcm"])
def test_provider_failure_retires_connection_and_next_request_reuses_engine(monkeypatch, phase):
    def failed(wire, flush):
        if phase == "after_pcm":
            wire.incoming.put(PCM)
        wire.incoming.put(json.dumps({"type": "error", "error": {"message": "provider boom"}}))

    engine, context, created = _engine(monkeypatch, [failed, _success])
    try:
        assert not engine.synthesize("first")
        first_error = engine.last_error
        assert first_error is not None and "provider boom" in str(first_error)
        if phase == "after_pcm":
            assert engine.queue.get(timeout=1) == PCM
        assert engine.synthesize("second")
        assert engine.queue.get(timeout=1) == PCM
        assert len(created) == 2
    finally:
        engine.close_stream_context(context)


def test_stop_cancels_waiting_provider_and_connection_can_be_reused(monkeypatch):
    entered = threading.Event()

    def blocked(wire, flush):
        del flush
        entered.set()

    engine, context, created = _engine(monkeypatch, [blocked, _success])
    result = []
    try:
        worker = threading.Thread(target=lambda: result.append(engine.synthesize("blocked")))
        worker.start()
        assert entered.wait(1)
        engine.stop()
        worker.join(1)
        assert not worker.is_alive()
        assert result == [False]
        assert engine.synthesize("after stop")
        assert engine.queue.get(timeout=1) == PCM
        assert len(created) == 2
    finally:
        engine.close_stream_context(context)


def test_cancel_after_connect_does_not_start_segment_and_connection_reuses(monkeypatch):
    engine, context, created = _engine(monkeypatch, [_success, _success])
    original_connect = remote.websocket.create_connection
    triggered = False

    def connect(*args, **kwargs):
        nonlocal triggered
        wire = original_connect(*args, **kwargs)
        if not triggered:
            triggered = True
            engine.stop()
        return wire

    monkeypatch.setattr(remote.websocket, "create_connection", connect)
    try:
        assert not engine.synthesize("cancelled after connect")
        assert engine.last_error is None
        assert [event["type"] for event in created[0].sent] == ["config", "cancel"]
        assert engine.synthesize("after cancellation")
        assert engine.queue.get(timeout=1) == PCM
        assert len(created) == 2
    finally:
        engine.close_stream_context(context)


def test_resume_uses_last_connection_after_fragment_completion(monkeypatch):
    engine, context, created = _engine(monkeypatch, [_success])
    try:
        assert engine.synthesize("one")
        assert engine.queue.get(timeout=1) == PCM
        engine.resume()
        assert created[0].sent[-1] == {"type": "resume"}
    finally:
        engine.close_stream_context(context)


def test_alignment_callback_failure_does_not_break_reuse(monkeypatch):
    def success_with_alignment(wire, flush):
        request_id = f"request-{flush['segment_id']}"
        wire.incoming.put(json.dumps({"type": "fragment_start", "request_id": request_id}))
        wire.incoming.put(PCM)
        wire.incoming.put(json.dumps({"type": "segment_done", "segment_id": flush["segment_id"]}))
        wire.incoming.put(json.dumps({"type": "alignment", "request_id": request_id, "words": []}))
        wire.incoming.put(json.dumps({"type": "alignment_done", "request_id": request_id, "complete": True}))

    engine, context, created = _engine(monkeypatch, [success_with_alignment], alignment=True)
    callback_seen = threading.Event()

    def failing_callback(_event):
        callback_seen.set()
        raise RuntimeError("test callback failure")

    engine.on_alignment = failing_callback
    try:
        assert engine.synthesize("one")
        assert engine.queue.get(timeout=1) == PCM
        assert callback_seen.wait(1)
        assert engine.synthesize("two")
        assert engine.queue.get(timeout=1) == PCM
        assert len(created) == 1
    finally:
        engine.close_stream_context(context)