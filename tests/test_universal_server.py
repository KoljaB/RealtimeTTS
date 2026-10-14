import asyncio
import json
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from RealtimeTTS._audio_backend import pyaudio
from RealtimeTTS.alignment.base import WordAlignment
from RealtimeTTS.server import create_app


RATE = 16_000


@pytest.mark.parametrize("event_type", ["done", "error", "cancelled", "alignment_done"])
def test_outbound_terminal_event_survives_full_audio_queue(event_type):
    from RealtimeTTS.server import _OutboundBuffer

    cancelled = threading.Event()
    outbound = _OutboundBuffer(cancelled, limit=1)
    assert outbound.audio(b"pcm")
    if event_type == "cancelled":
        cancelled.set()

    terminal = {"type": event_type, "request_id": "request"}
    assert outbound.event(terminal)
    assert outbound.get(timeout=0) == ("audio", b"pcm")
    assert outbound.get(timeout=0) == ("event", terminal)
    assert outbound.empty()


def test_outbound_terminal_overflow_stays_bounded_and_ordered():
    from RealtimeTTS.server import _OutboundBuffer

    outbound = _OutboundBuffer(threading.Event(), limit=1)
    assert outbound.audio(b"pcm")
    event_types = ["alignment_done", "error", "cancelled", "done"]
    for revision in range(10):
        for event_type in event_types:
            assert outbound.event({"type": event_type, "revision": revision})

    assert outbound.get(timeout=0) == ("audio", b"pcm")
    for event_type in event_types:
        assert outbound.get(timeout=0) == (
            "event", {"type": event_type, "revision": 9}
        )
    with pytest.raises(queue.Empty):
        outbound.get(timeout=0)


def test_outbound_terminal_reserve_does_not_expand_audio_capacity():
    from RealtimeTTS.server import _OutboundBuffer

    cancelled = threading.Event()
    outbound = _OutboundBuffer(cancelled, limit=1)
    assert outbound.audio(b"first")
    assert outbound.event({"type": "alignment_done"})
    entered = threading.Event()
    finished = threading.Event()
    accepted = []

    def produce():
        entered.set()
        accepted.append(outbound.audio(b"second"))
        finished.set()

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    try:
        assert entered.wait(1)
        assert not finished.wait(0.05)
        assert outbound.get(timeout=0) == ("audio", b"first")
        assert not finished.wait(0.05)
        assert outbound.get(timeout=0) == ("event", {"type": "alignment_done"})
        assert finished.wait(1)
        assert accepted == [True]
        assert outbound.get(timeout=0) == ("audio", b"second")
    finally:
        cancelled.set()
        producer.join(timeout=1)
    assert not producer.is_alive()


def test_outbound_full_audio_queue_remains_cancellable():
    from RealtimeTTS.server import _OutboundBuffer

    cancelled = threading.Event()
    outbound = _OutboundBuffer(cancelled, limit=1)
    assert outbound.audio(b"first")
    finished = threading.Event()
    accepted = []

    def produce():
        accepted.append(outbound.audio(b"second"))
        finished.set()

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    try:
        assert not finished.wait(0.05)
        cancelled.set()
        assert finished.wait(1)
        assert accepted == [False]
    finally:
        cancelled.set()
        producer.join(timeout=1)
    assert not producer.is_alive()


class FakeEngine:
    engine_name = "fake-generic"
    model_id = "fake-model"
    device = "cpu"

    def __init__(self, *, block=False, samples=160, native=False):
        self.queue = queue.Queue()
        self.timings = queue.Queue()
        self.audio_duration = 0.0
        self.provides_word_timings = bool(native)
        self.current_voice = None
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.release = threading.Event()
        self.block = block
        self.samples = int(samples)

    def get_stream_info(self):
        return pyaudio.paInt16, 1, RATE

    def get_voices(self):
        return ["default"]

    def set_voice(self, voice):
        self.current_voice = voice

    def synthesize(self, text):
        self.started.set()
        if self.block:
            self.release.wait(3)
        if self.stopped.is_set():
            return False
        samples = np.full(self.samples, 1_500, dtype="<i2")
        self.queue.put(samples.tobytes())
        self.audio_duration += len(samples) / RATE
        if self.provides_word_timings:
            self.timings.put(
                SimpleNamespace(word=text.strip(), start_time=0.0, end_time=0.01)
            )
        return bool(text.strip())

    def stop(self):
        self.stopped.set()
        self.release.set()


def test_cancelled_http_request_restores_engine_and_allows_next_request():
    import httpx

    async def exercise():
        engine = FakeEngine(block=True)
        original_queue, original_timings = engine.queue, engine.timings
        app = create_app(engine)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            request = asyncio.create_task(
                client.post("/v1/audio/speech", json={"input": "cancel me"})
            )
            try:
                assert await asyncio.to_thread(engine.started.wait, 1)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                assert engine.stopped.is_set()
                assert engine.queue is original_queue
                assert engine.timings is original_timings
                assert app.state.realtimetts_server.health()["active_requests"] == 0

                engine.block = False
                engine.stopped.clear()
                response = await client.post(
                    "/v1/audio/speech", json={"input": "next request"}
                )
                assert response.status_code == 200
                assert response.content
            finally:
                engine.release.set()
                if not request.done():
                    request.cancel()
                try:
                    await request
                except asyncio.CancelledError:
                    pass

    asyncio.run(exercise())


def test_cancelling_completed_session_does_not_stop_new_request():
    engine = FakeEngine()
    original_queue, original_timings = engine.queue, engine.timings
    host = create_app(engine).state.realtimetts_server
    completed = host.new_session({})
    completed.start()
    completed.add_text("first")
    completed.end()
    assert completed.done.wait(1)

    engine.block = True
    engine.started.clear()
    active = host.new_session({})
    active.start()
    active.add_text("second")
    active.end()
    try:
        assert engine.started.wait(1)
        completed.cancel()
        assert not engine.stopped.is_set()
        assert not active.done.is_set()
    finally:
        active.cancel()
        assert active.done.wait(1)
        active._worker.join(timeout=1)
    assert not active._worker.is_alive()
    assert engine.queue is original_queue
    assert engine.timings is original_timings


class FixedAligner:
    supports_partial = False

    def align_words(self, text, audio, sample_rate):
        del audio, sample_rate
        return [WordAlignment(text.strip(), 0.0, 0.01)]


def _receive_until_done(ws):
    events = []
    audio = []
    for _ in range(100):
        message = ws.receive()
        if message.get("type") == "websocket.close":
            break
        if message.get("bytes") is not None:
            audio.append(message["bytes"])
            continue
        raw = message.get("text")
        if raw is None:
            continue
        event = json.loads(raw)
        events.append(event)
    return events, b"".join(audio)


def test_generic_profile_assets_health_and_capabilities():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/studio/studio.js").status_code == 200
        assert client.get("/health").json()["status"] == "ok"
        capabilities = client.get("/v1/capabilities").json()
        assert capabilities["protocol"] == "realtimetts-generic-http"
        assert capabilities["audio"] == {
            "encoding": "s16le",
            "sample_rate_hz": 24_000,
            "channels": 1,
            "bits_per_sample": 16,
            "response_formats": ["pcm", "wav"],
            "pcm_media_type": "audio/pcm",
            "wav_media_type": "audio/wav",
        }
        assert capabilities["features"]["model_type"] == "generic"
        assert capabilities["stream_controls"]["native_pause_resume"] is False
        assert "pause" not in capabilities["stream_controls"]["input_events"]
        assert client.get("/v1/models").json()["data"][0]["id"] == "fake-model"
        assert client.get("/v1/audio/voices").json()["voices"][0]["name"] == "default"
        assert client.get("/health").json()["active_requests"] == 0


def test_http_speech_is_resampled_to_fixed_s16_pcm():
    engine = FakeEngine(samples=160)
    with TestClient(create_app(engine)) as client:
        response = client.post(
            "/v1/audio/speech",
            json={"input": "hello", "voice": "default", "response_format": "pcm"},
        )
        assert response.status_code == 200, response.text
        assert response.headers["x-audio-encoding"] == "s16le"
        samples = np.frombuffer(response.content, dtype="<i2")
        assert len(samples) > 160
        assert np.max(np.abs(samples)) > 100


def test_websocket_stream_includes_alignment_snapshots_and_s16_audio():
    engine = FakeEngine(samples=240)
    with TestClient(create_app(engine, aligner=FixedAligner())) as client:
        with client.websocket_connect("/v1/audio/speech-stream", subprotocols=["qwen-studio"]) as ws:
            ws.send_json({
                "type": "config",
                "voice": "default",
                "alignment": {"source": "auto", "level": "word", "mode": "live"},
            })
            ws.send_json({"type": "text", "text": "hello"})
            ws.send_json({"type": "end"})
            events, audio = _receive_until_done(ws)

    assert events[0]["type"] == "ready"
    assert any(event["type"] == "fragment_start" for event in events)
    alignments = [event for event in events if event["type"] == "alignment"]
    assert alignments
    assert any(item["word"] == "hello" for event in alignments for item in event["words"])
    assert any(event["type"] == "alignment_done" for event in events)
    assert events[-1]["type"] == "done"
    decoded = np.frombuffer(audio, dtype="<i2")
    assert len(decoded) > 0
    assert np.max(np.abs(decoded)) > 100


def test_partial_alignment_remains_partial_in_terminal_websocket_event():
    class PrefixOnly:
        def align_words(self, text, audio, sample_rate):
            return [WordAlignment("hello", 0, .005, .9)]

    with TestClient(create_app(FakeEngine(), aligner=PrefixOnly())) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({"type": "config", "alignment": {"source": "model", "mode": "export"}})
            ws.send_json({"type": "text", "text": "hello world"})
            ws.send_json({"type": "end"})
            events, audio = _receive_until_done(ws)
    done = next(event for event in events if event["type"] == "alignment_done")
    assert audio and done["final"]
    assert done["status"] == "partial" and done["complete"] is False
    assert done["coverage"] == .5
    assert done["fragments"][0]["status"] == "partial"


def test_alignment_requires_explicit_aligner_or_native_engine():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({
                "type": "config",
                "alignment": {"source": "auto", "level": "word"},
            })
            event = json.loads(ws.receive_text())
    assert event["type"] == "error"
    assert event["error"]["type"] == "unsupported_operation"


def test_cancel_stops_a_running_engine_and_reports_cancelled():
    engine = FakeEngine(block=True)
    with TestClient(create_app(engine)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({"type": "config", "voice": "default"})
            ws.send_json({"type": "text", "text": "long"})
            assert engine.started.wait(2)
            ws.send_json({"type": "cancel"})
            events, _ = _receive_until_done(ws)
    assert engine.stopped.is_set()
    assert any(event["type"] == "cancelled" for event in events)


def test_output_limit_is_bounded_and_returns_error():
    engine = FakeEngine(samples=1_000)
    with TestClient(create_app(engine, max_output_bytes=64)) as client:
        response = client.post(
            "/v1/audio/speech",
            json={"input": "bounded", "voice": "default"},
        )
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "output_error"


def test_api_key_protects_inference_but_keeps_studio_public():
    engine = FakeEngine()
    with TestClient(create_app(engine, api_key="secret")) as client:
        assert client.get("/").status_code == 200
        assert client.get("/v1/capabilities").status_code == 401
        assert client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer secret"}
        ).status_code == 200


def test_native_auto_uses_engine_word_timings_without_model():
    engine = FakeEngine(native=True)
    with TestClient(create_app(engine)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({
                "type": "config",
                "request_id": "native-1",
                "alignment": {"source": "auto", "level": "word"},
            })
            ws.send_json({"type": "text", "text": "native"})
            ws.send_json({"type": "end"})
            events, _ = _receive_until_done(ws)
    assert any(event["type"] == "alignment" for event in events)
    assert not any(
        event["type"] == "error" and event["error"]["type"] == "unsupported_operation"
        for event in events
    )


def test_websocket_orders_each_fragment_boundary_around_pcm():
    engine = FakeEngine(samples=240)
    with TestClient(create_app(engine, aligner=FixedAligner())) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({
                "type": "config",
                "request_id": "ordered-1",
                "alignment": {"source": "auto", "level": "word"},
            })
            ws.send_json({"type": "text", "text": "first"})
            ws.send_json({"type": "text", "text": "second"})
            ws.send_json({"type": "end"})
            messages = []
            for _ in range(100):
                message = ws.receive()
                if message.get("type") == "websocket.close":
                    break
                messages.append(message)
    sequence = []
    for message in messages:
        if message.get("bytes") is not None:
            sequence.append(("audio", None))
        elif message.get("text"):
            event = json.loads(message["text"])
            sequence.append((event["type"], event))
    first_start = next(i for i, item in enumerate(sequence) if item[0] == "fragment_start" and item[1]["fragment_index"] == 0)
    first_end = next(i for i, item in enumerate(sequence) if item[0] == "fragment_end" and item[1]["fragment_index"] == 0)
    second_start = next(i for i, item in enumerate(sequence) if item[0] == "fragment_start" and item[1]["fragment_index"] == 1)
    second_end = next(i for i, item in enumerate(sequence) if item[0] == "fragment_end" and item[1]["fragment_index"] == 1)
    assert any(item[0] == "audio" for item in sequence[first_start + 1:first_end])
    assert first_end < second_start < second_end
    assert all(
        item[1].get("request_id") == "ordered-1"
        for item in sequence
        if item[1] is not None and item[0] in {"fragment_start", "fragment_end", "alignment_done", "done"}
    )
    alignment_events = [item[1] for item in sequence if item[0] == "alignment"]
    assert alignment_events
    assert all(event["time_base"] == "fragment" for event in alignment_events)
    assert all(
        span["start_time"] <= span["end_time"] <= 0.02
        for event in alignment_events
        for span in event["words"]
    )


def test_cancel_without_text_finishes():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({"type": "config"})
            ws.send_json({"type": "cancel"})
            events, _ = _receive_until_done(ws)
    assert any(event["type"] == "cancelled" for event in events)


def test_api_rejects_ws_only_alignment_and_unsafe_stream_config():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        response = client.post(
            "/v1/audio/speech",
            json={"input": "hello", "alignment": True},
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "unsupported_operation"
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({"type": "config", "response_format": "wav"})
            event = json.loads(ws.receive_text())
        assert event["error"]["type"] == "invalid_request_error"
        for alignment in ({"model_dir": "C:/private"}, {"max_buffer_seconds": 2.0}):
            with client.websocket_connect("/v1/audio/speech-stream") as ws:
                ws.send_json({"type": "config", "alignment": alignment})
                event = json.loads(ws.receive_text())
            assert event["error"]["type"] == "invalid_request_error"
