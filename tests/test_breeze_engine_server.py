import base64
import io
import json
import queue
import threading
import wave
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from RealtimeTTS import BreezeEngine, BreezeVoice
from RealtimeTTS.engines.breeze_engine import (
    BreezeEngineError,
    directed_blocks,
    strip_inline_vocal_tags,
)
from RealtimeTTS.breeze_server import BreezeHttpServer
from RealtimeTTS.qwen_server import create_app


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\xe8\x03" * 2400)
    return output.getvalue()


def test_directions_strip_only_paragraph_tags_and_whitespace():
    assert list(directed_blocks(
        "[Amused]\r\n \t\n Hello (laugh).\r\n\r\n[Whispering softly]\r\n\tNow (breathe heavily) listen.",
        "Natural",
    )) == [
        ("Hello (laugh).", "Amused"),
        ("Now (breathe heavily) listen.", "Whispering softly"),
    ]
    assert list(directed_blocks("ordinary [quoted] text (chuckle)", "Warm")) == [
        ("ordinary [quoted] text (chuckle)", "Warm")
    ]


class Response:
    def __init__(self, payload):
        self.payload = payload
    def raise_for_status(self):
        pass
    def json(self):
        return self.payload


class Http:
    def __init__(self):
        self.posts = []
    def get(self, url, **kwargs):
        if url.endswith("/health"):
            return Response({"sample_rate": 24000, "native_checkpoints": True})
        return Response([{"id": "test", "ref_text": "reference"}])
    def post(self, url, **kwargs):
        self.posts.append(kwargs)
        return Response({"id": kwargs["data"]["name"]})
    def close(self):
        pass


class Socket:
    def __init__(self, *, hold=False):
        self.messages = []
        self.incoming = queue.Queue()
        self.incoming.put(json.dumps({"type": "ready"}))
        self.hold = hold
        self.text_received = threading.Event()
    def send(self, raw):
        message = json.loads(raw)
        self.messages.append(message)
        event = message["type"]
        if event == "start":
            self.incoming.put(json.dumps({"type": "started"}))
        elif event == "text":
            self.text_received.set()
            self.incoming.put(b"\xe8\x03" * 200)
        elif event == "end" and not self.hold:
            self.incoming.put(json.dumps({"type": "done"}))
        elif event in {"pause", "resume", "cancel"}:
            self.incoming.put(json.dumps({"type": {"pause":"paused", "resume":"resumed", "cancel":"cancelled"}[event]}))
    def recv(self):
        return self.incoming.get(timeout=2)
    def close(self, **kwargs):
        pass
    def shutdown(self):
        self.close()


def make_engine(monkeypatch, socket=None):
    import RealtimeTTS.engines.breeze_engine as module
    http = Http()
    monkeypatch.setattr(module.requests, "Session", lambda: http)
    if socket is not None:
        monkeypatch.setattr(module.websocket, "create_connection", lambda *a, **k: socket)
    return BreezeEngine(BreezeVoice("test", voice_id="test")), http


def test_engine_translates_registered_vocalizations_from_native_tags(monkeypatch):
    engine, _ = make_engine(monkeypatch)
    try:
        assert engine.get_vocalization_text("sigh", instructions="Gentle sigh") == "(sigh)"
        assert engine.get_vocalization_text("clear_throat") == "(clear throat)"
        with pytest.raises(NotImplementedError, match="vocalization"):
            engine.get_vocalization_text("unsupported_sound")
    finally:
        engine.shutdown()


def test_public_engine_streams_pcm_and_directs_without_reading_tags(monkeypatch):
    sock = Socket()
    engine, _ = make_engine(monkeypatch, sock)
    assert engine.get_stream_info() == (8, 1, 24000)
    assert engine.synthesize("[Amused]\r\n\n Hello (laugh).")
    assert engine.queue.get_nowait() == b"\xe8\x03" * 200
    assert next(m for m in sock.messages if m["type"] == "start")["instruction"] == "Amused"
    assert next(m for m in sock.messages if m["type"] == "text")["text"] == "Hello (laugh)."
    assert engine.get_voices()[0].voice_id == "test"


def test_reference_prefix_cache_requires_health_and_advertises_payload(
    monkeypatch, tmp_path
):
    import RealtimeTTS.engines.breeze_engine as module

    with pytest.raises(BreezeEngineError, match="reference prefix cache"):
        http = Http()
        monkeypatch.setattr(module.requests, "Session", lambda: http)
        BreezeEngine(
            BreezeVoice("test", voice_id="test"),
            reference_prefix_cache=True,
        )

    class CacheHttp(Http):
        def get(self, url, **kwargs):
            response = super().get(url, **kwargs)
            if url.endswith("/health"):
                response.payload["reference_prefix_cache"] = True
            return response

    http = CacheHttp()
    sock = Socket()
    monkeypatch.setattr(module.requests, "Session", lambda: http)
    monkeypatch.setattr(module.websocket, "create_connection", lambda *a, **k: sock)
    engine = BreezeEngine(
        BreezeVoice("test", voice_id="test"),
        reference_prefix_cache=True,
    )
    assert engine.synthesize("Cache the reference prefix.")
    start = next(m for m in sock.messages if m["type"] == "start")
    assert start["cache_reference_prefix"] is True

    server = BreezeHttpServer(engine, voice_dir=tmp_path / "voices")
    assert server.capabilities()["features"]["reference_prefix_cache"] is True
    engine.shutdown()


def test_voice_cache_tracks_audio_and_transcript(monkeypatch, tmp_path):
    engine, http = make_engine(monkeypatch)
    reference = tmp_path / "reference.wav"
    reference.write_bytes(wav_bytes())
    voice = BreezeVoice("custom", reference, "Exact transcript.")
    engine.set_voice(voice)
    engine.set_voice(voice)
    assert len(http.posts) == 1
    engine.set_voice(BreezeVoice("custom", reference, "Changed transcript."))
    assert len(http.posts) == 2
    assert http.posts[0]["data"]["name"] != http.posts[1]["data"]["name"]
    with pytest.raises(BreezeEngineError, match="transcript"):
        engine.set_voice(BreezeVoice("bad", reference))


def test_pause_cancel_waits_for_native_ack_and_drops_late_audio(monkeypatch):
    sock = Socket(hold=True)
    engine, _ = make_engine(monkeypatch, sock)
    worker = threading.Thread(target=engine.synthesize, args=("This is a long sentence.",))
    worker.start()
    assert sock.text_received.wait(2)
    assert engine.pause(timeout=1)
    engine.resume()
    engine.stop()
    worker.join(2)
    assert not worker.is_alive()
    assert engine.last_metrics["cancelled"]
    assert any(m["type"] == "cancel" for m in sock.messages)


def test_cancel_before_synthesize_does_not_start_native_work(monkeypatch):
    sock = Socket()
    engine, _ = make_engine(monkeypatch, sock)
    cancelled = threading.Event()
    cancelled.set()
    engine.queue.state = SimpleNamespace(cancelled=cancelled)
    assert not engine.synthesize("Must never reach the worker.")
    assert not sock.messages



def test_server_can_start_without_a_reference_and_synthesize_voice_design(monkeypatch, tmp_path):
    import uvicorn
    import RealtimeTTS.breeze_server as module

    engine = FakeBreeze()
    monkeypatch.setattr(module, "BreezeEngine", lambda **kwargs: engine)
    def serve(app, **kwargs):
        with TestClient(app) as client:
            assert client.get("/ready").status_code == 200
            response = client.post("/v1/audio/speech", json={
                "input": "A fresh start.", "voice": module.DESIGN_VOICE,
                "instructions": "A calm adult narrator.", "response_format": "pcm",
            })
            assert response.status_code == 200, response.text
            assert response.content == b"\xe8\x03" * 1200
    monkeypatch.setattr(uvicorn, "run", serve)
    module.main(["--worker-url", "http://127.0.0.1:8137",
                 "--no-bundled-voices", "--voice-dir", str(tmp_path / "voices")])
    assert engine.seen == [("A fresh start.", "A calm adult narrator.")]


class FakeBreeze:
    direct_voice_control = instruction_control = True
    engine_name = "breeze"
    model_type = "base"
    cfg_scale = 4.0
    seed = 44
    max_new_tokens = 500
    def __init__(self):
        self.queue = queue.Queue()
        self.seen = []
    def set_voice(self, voice):
        self.voice = voice
    def set_voice_parameters(self, **kwargs):
        self.parameters = kwargs
    def synthesize(self, text):
        self.seen.append((text, self.voice.instruct))
        self.queue.put(b"\xe8\x03" * 1200)
        return True
    def stop(self):
        pass
    def resume(self):
        pass
    def shutdown(self):
        pass


def server_client(tmp_path):
    engine = FakeBreeze()
    server = BreezeHttpServer(engine, voice_dir=tmp_path / "voices")
    server.register_voice({"name":"test", "ref_text":"exact reference",
                           "wav_b64":base64.b64encode(wav_bytes()).decode("ascii")})
    return engine, TestClient(create_app(server))


def test_server_contract_and_sticky_per_text_instructions(tmp_path):
    engine, client = server_client(tmp_path)
    with client:
        caps = client.get("/v1/capabilities").json()
        assert caps["protocol"] == "realtimetts-qwen-http"
        assert caps["features"]["direct_voice_control"]
        assert caps["engine"]["backend"] == "Breeze-TTS-2.cpp"
        assert "breathe heavily" in caps["features"]["vocal_tags"]
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({"type":"config", "voice":"test", "model":"breeze-tts-2-q8", "response_format":"pcm"})
            ws.send_json({"type":"text", "instructions":"Amused", "text":"Hello (chuckle), this is funny."})
            ws.send_json({"type":"text", "instructions":"Whisper softly", "text":"\r\n \t\nNow (breathe heavily) listen."})
            ws.send_json({"type":"end"})
            events, pcm = [], b""
            while True:
                message = ws.receive()
                if message.get("bytes"):
                    pcm += message["bytes"]
                elif message.get("text"):
                    event = json.loads(message["text"])
                    events.append(event["type"])
                    if event["type"] in {"done", "cancelled", "error"}:
                        break
            assert events[-1] == "done", (events, message)
            assert pcm
        warm = " ".join(text for text, ins in engine.seen if ins == "Amused")
        whisper = " ".join(text for text, ins in engine.seen if ins == "Whisper softly")
        assert "(chuckle)" in warm and "funny." in warm
        assert "Now (breathe heavily) listen." in whisper
        assert all("Whisper softly" not in text for text, _ in engine.seen)


@pytest.mark.parametrize("extra", [{"clone_mode":"speaker_only"}, {"cfg_scale":0}, {"cfg_scale":True}])
def test_reject_unsupported_or_invalid_controls(tmp_path, extra):
    _, client = server_client(tmp_path)
    with client:
        response = client.post("/v1/audio/speech", json={"voice":"test", "input":"hello", **extra})
        assert response.status_code == 400


class FakeAlignmentSession:
    def __init__(self, callback):
        self.callback = callback
        self.feed_samples = 0
        self.finished = None
        self.stop_requested = threading.Event()
        self.finish_reason = None

    def feed(self, audio):
        self.feed_samples += len(audio)

    def finish(self, *, cancelled=False):
        self.finished = cancelled
        self.callback({
            "type": "alignment",
            "words": [{"word": "simple", "start_time": 0.1, "end_time": 0.3}],
            "characters": [],
            "complete": True,
        })
        self.callback({
            "type": "alignment_done",
            "complete": True,
            "cancelled": cancelled,
        })


class FakeAligner:
    timestamp_levels = ("word", "character")
    language_timestamps = {"english": ["word", "character"]}

    def __init__(self):
        self.backend = SimpleNamespace(language="english")
        self.calls = []
        self.session = None
        self.closed = False

    def start(self, text, sample_rate, **kwargs):
        self.calls.append({"text": text, "sample_rate": sample_rate, **kwargs})
        self.session = FakeAlignmentSession(kwargs["callback"])
        return self.session

    def close(self):
        self.closed = True


class LateCallbackSession:
    def __init__(self, owner, index, callback):
        self.owner = owner
        self.index = index
        self.callback = callback
        self.finished = None
        self.stop_requested = threading.Event()

    def feed(self, audio):
        pass

    def finish(self, *, cancelled=False):
        self.finished = cancelled
        if self.index == 0:
            # The worker can publish a final update after finish() returns.
            self.owner.pending = self.callback
            return
        self.callback({
            "type": "alignment",
            "words": [{"word": "second", "start_time": 0.2, "end_time": 0.4}],
            "characters": [],
            "complete": True,
        })
        self.callback({
            "type": "alignment_done",
            "complete": True,
            "cancelled": cancelled,
        })


class LateCallbackAligner(FakeAligner):
    def __init__(self):
        super().__init__()
        self.sessions = []
        self.pending = None

    def start(self, text, sample_rate, **kwargs):
        self.calls.append({"text": text, "sample_rate": sample_rate, **kwargs})
        session = LateCallbackSession(self, len(self.sessions), kwargs["callback"])
        self.sessions.append(session)
        if self.pending is not None:
            callback, self.pending = self.pending, None
            callback({
                "type": "alignment",
                "words": [{"word": "first", "start_time": 0.1, "end_time": 0.3}],
                "characters": [],
                "complete": True,
            })
        return session


class PostReturnSession:
    def __init__(self, callback):
        self.callback = callback
        self.finished = None
        self.stop_requested = threading.Event()

    def feed(self, audio):
        pass

    def finish(self, *, cancelled=False):
        self.finished = cancelled

    def emit_final(self):
        self.callback({
            "type": "alignment",
            "words": [{"word": "final", "start_time": 0.15, "end_time": 0.35}],
            "characters": [],
            "complete": True,
        })
        self.callback({
            "type": "alignment_done",
            "complete": True,
            "cancelled": False,
        })


class PostReturnAligner(FakeAligner):
    def start(self, text, sample_rate, **kwargs):
        self.calls.append({"text": text, "sample_rate": sample_rate, **kwargs})
        self.session = PostReturnSession(kwargs["callback"])
        return self.session


def test_alignment_ranges_are_utf16_and_ignore_tone_directions():
    text = "A😀 (laugh) and (breathe heavily)."
    assert strip_inline_vocal_tags(text) == "A😀   and  ."
    assert BreezeEngine.alignment_non_speech_ranges(text) == [
        {"start": 4, "end": 11},
        {"start": 16, "end": 33},
    ]
    assert BreezeEngine.alignment_non_speech_ranges("[Whispering] hello") == []


def test_alignment_starts_after_pcm_and_preserves_vocal_insert(monkeypatch):
    import RealtimeTTS.engines.breeze_engine as module

    http = Http()
    sock = Socket()
    aligner = FakeAligner()
    monkeypatch.setattr(module.requests, "Session", lambda: http)
    monkeypatch.setattr(module.websocket, "create_connection", lambda *a, **k: sock)
    engine = BreezeEngine(
        BreezeVoice("test", voice_id="test"),
        streaming_aligner=aligner,
    )
    gate = threading.Event()
    engine.alignment_start_event = gate

    assert engine.synthesize("A (laugh) simple line.")
    assert engine.queue.get_nowait() == b"\xe8\x03" * 200
    assert not gate.is_set()
    assert aligner.calls[0]["text"] == "A   simple line."
    assert aligner.calls[0]["stop_on_completion"] is False
    assert aligner.calls[0]["language"] == "english"
    assert aligner.session.feed_samples == 200
    assert aligner.session.finished is False
    timing = engine.timings.get_nowait()
    assert timing.word == "simple"
    assert timing.start_time == pytest.approx(0.1)
    assert timing.end_time == pytest.approx(0.3)
    engine.shutdown()
    assert not aligner.closed


def test_alignment_cancel_finishes_session_without_cutting_pcm(monkeypatch):
    import RealtimeTTS.engines.breeze_engine as module

    http = Http()
    sock = Socket(hold=True)
    aligner = FakeAligner()
    monkeypatch.setattr(module.requests, "Session", lambda: http)
    monkeypatch.setattr(module.websocket, "create_connection", lambda *a, **k: sock)
    engine = BreezeEngine(
        BreezeVoice("test", voice_id="test"),
        streaming_aligner=aligner,
    )
    worker = threading.Thread(target=engine.synthesize, args=("A (chuckle) long line.",))
    worker.start()
    assert sock.text_received.wait(2)
    engine.stop()
    worker.join(2)
    assert not worker.is_alive()
    assert engine.last_metrics["cancelled"]
    assert aligner.session.finished is True
    assert aligner.session.feed_samples == 200
    engine.shutdown()


def test_alignment_model_configuration_passes_cpu_threads(monkeypatch):
    import RealtimeTTS.alignment as alignment_module
    import RealtimeTTS.engines.breeze_engine as module

    http = Http()
    monkeypatch.setattr(module.requests, "Session", lambda: http)
    seen = {}

    class ConstructedAligner(FakeAligner):
        def __init__(self, model, **kwargs):
            super().__init__()
            seen["model"] = model
            seen.update(kwargs)

    monkeypatch.setattr(alignment_module, "StreamingTextAligner", ConstructedAligner)
    engine = BreezeEngine(
        BreezeVoice("test", voice_id="test"),
        alignment_model="alignment-model",
        alignment_language="english",
        alignment_threads=3,
    )
    assert seen == {"model": "alignment-model", "language": "english", "threads": 3}
    assert engine.provides_word_timings
    engine.shutdown()
    assert engine.streaming_aligner is None


def test_device_uses_worker_backend_health(monkeypatch):
    import RealtimeTTS.engines.breeze_engine as module

    class BackendHttp(Http):
        def get(self, url, **kwargs):
            response = super().get(url, **kwargs)
            if url.endswith("/health"):
                response.payload["backend"] = "CUDA0"
            return response

    http = BackendHttp()
    monkeypatch.setattr(module.requests, "Session", lambda: http)
    engine = BreezeEngine(BreezeVoice("test", voice_id="test"))
    assert engine.device == "cuda0"
    engine.shutdown()


def test_late_previous_block_callback_keeps_original_timing(monkeypatch):
    import RealtimeTTS.engines.breeze_engine as module

    http = Http()
    sockets = [Socket(), Socket()]
    aligner = LateCallbackAligner()
    monkeypatch.setattr(module.requests, "Session", lambda: http)
    monkeypatch.setattr(
        module.websocket,
        "create_connection",
        lambda *a, **k: sockets.pop(0),
    )
    engine = BreezeEngine(
        BreezeVoice("test", voice_id="test"),
        streaming_aligner=aligner,
    )
    initial_generation = engine._alignment_generation
    assert initial_generation == 1

    assert engine.synthesize("[First]\nfirst line.\n\n[Second]\nsecond line.")
    timings = [engine.timings.get_nowait(), engine.timings.get_nowait()]
    with pytest.raises(queue.Empty):
        engine.timings.get_nowait()
    assert timings[0].word == "first"
    assert timings[0].start_time == pytest.approx(0.1)
    assert timings[1].word == "second"
    assert timings[1].start_time == pytest.approx(0.008333333 + 0.2)
    assert aligner.sessions[0].finished is False
    assert aligner.sessions[1].finished is False

    engine.reset_audio_duration()
    assert engine._alignment_generation == initial_generation + 1
    engine.shutdown()


def test_final_alignment_callback_after_synthesize_return(monkeypatch):
    import RealtimeTTS.engines.breeze_engine as module

    http = Http()
    sock = Socket()
    aligner = PostReturnAligner()
    events = []
    monkeypatch.setattr(module.requests, "Session", lambda: http)
    monkeypatch.setattr(module.websocket, "create_connection", lambda *a, **k: sock)
    engine = BreezeEngine(
        BreezeVoice("test", voice_id="test"),
        streaming_aligner=aligner,
        on_alignment=events.append,
    )

    assert engine.synthesize("A final line.")
    assert aligner.session.finished is False
    aligner.session.emit_final()
    timing = engine.timings.get_nowait()
    assert timing.word == "final"
    assert timing.start_time == pytest.approx(0.15)
    assert events[-1]["type"] == "alignment_done"

    event_count = len(events)
    engine.reset_audio_duration()
    aligner.session.emit_final()
    with pytest.raises(queue.Empty):
        engine.timings.get_nowait()
    assert len(events) == event_count
    engine.shutdown()

@pytest.mark.parametrize("text", ["(chuckle)", "(laugh) (sigh)"])
def test_vocal_only_fragment_completes_alignment_without_running_asr(monkeypatch, text):
    engine, _ = make_engine(monkeypatch, Socket())
    aligner = FakeAligner()
    engine.streaming_aligner = aligner
    # Reproduce the voice-only fragment following an already aligned sentence.
    engine.last_alignment = object()
    events = []
    engine.on_alignment = events.append
    assert engine.synthesize(text)
    assert engine.queue.get_nowait() == b"\xe8\x03" * 200
    assert not aligner.calls
    assert events == [{"type": "alignment_done", "complete": True}]
    engine.shutdown()
