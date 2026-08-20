import asyncio
import base64
import io
import json
import queue
import threading
import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

import RealtimeTTS.qwen_server as qwen_server_module
from RealtimeTTS.qwen_server import (
    ApiError,
    QwenHttpServer,
    RequestMetrics,
    SpeechRequest,
    VoiceRegistry,
    _resolve_api_key,
    _validate_bind_security,
    build_argument_parser,
    create_app,
)
from RealtimeTTS.language_router import LanguageDetection, QwenLanguageRouter


class FakeEngine:
    def __init__(self):
        self.queue = queue.Queue()
        self.seed = -1
        self.max_new_tokens = 2048
        self.temperature = 0.9
        self.top_k = 50
        self.top_p = 1.0
        self.repetition_penalty = 1.05
        self.current_voice = None
        self.last_error = None
        self.parameters = {}
        self.stopped = False
        self.shutdown_called = False
        self.warmup_calls = 0
        self.silent = (np.zeros(16, dtype="<i2")).tobytes()
        self.audible = (np.array([1000, -1000] * 32, dtype="<i2")).tobytes()

    def set_voice(self, voice):
        if voice.spk_path:
            assert voice.spk_path.is_file()
            assert voice.rvq_path.is_file()
        else:
            assert voice.ref_audio.is_file()
        self.current_voice = voice

    def set_voice_parameters(self, **parameters):
        self.parameters = dict(parameters)
        for name, value in parameters.items():
            setattr(self, name, value)

    def synthesize(self, text):
        self.last_error = None
        if text == "failure":
            self.last_error = RuntimeError("native synthesis failed")
            return False
        if text == "empty":
            self.last_error = RuntimeError(
                "QwenEngine failed while streaming synthesis: "
                "qwentts.cpp completed without producing audio"
            )
            return False
        if text == "silent":
            self.queue.put(self.silent)
            return True
        self.queue.put(self.silent)
        self.queue.put(self.audible)
        return True

    def warmup(self):
        self.warmup_calls += 1
        self.queue.put(b"hidden warmup audio")

    def stop(self):
        self.stopped = True

    def shutdown(self):
        self.shutdown_called = True


class BlockingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def synthesize(self, text):
        self.entered.set()
        self.release.wait(2.0)
        return super().synthesize(text)


class LateChunkEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.first_chunk = threading.Event()
        self.release = threading.Event()

    def synthesize(self, _text):
        self.queue.put(self.silent)
        self.first_chunk.set()
        self.release.wait(2.0)
        self.queue.put(self.audible)
        return True


def _server(tmp_path, engine=None, **kwargs):
    return QwenHttpServer(
        engine or FakeEngine(),
        voice_dir=tmp_path / "voices",
        **kwargs,
    )


def _voice_payload(name="mira", ref_text="exact reference"):
    return {
        "name": name,
        "ref_text": ref_text,
        "spk_b64": base64.b64encode(b"speaker latents").decode("ascii"),
        "rvq_b64": base64.b64encode(b"codec latents").decode("ascii"),
    }


def _register(client, name="mira", ref_text="exact reference"):
    response = client.post(
        "/v1/audio/voices",
        json=_voice_payload(name=name, ref_text=ref_text),
    )
    assert response.status_code == 200, response.text


def _persist_voice(tmp_path, name="mira", ref_text="exact reference"):
    registry = VoiceRegistry(tmp_path / "voices")
    record = registry.candidate(
        name,
        ref_text,
        "latents",
        (b"speaker latents", b"codec latents"),
    )
    registry.commit(record)


def _wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def _direct_request(server):
    return server.parse_speech(
        {"input": "audible", "voice": "mira", "response_format": "pcm"}
    )


def test_protocol_capabilities_readiness_ids_and_audio_metadata(tmp_path):
    engine = FakeEngine()
    server = _server(tmp_path, engine)
    with TestClient(create_app(server)) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ready"}

        capabilities = client.get("/v1/capabilities")
        assert capabilities.status_code == 200
        payload = capabilities.json()
        assert payload["protocol_version"] == 1
        assert payload["server_version"] == qwen_server_module._server_version()
        assert payload["server_version"] != "unknown"
        assert payload["engine"]["backend"] == "qwentts.cpp"
        assert payload["engine"]["device"] == "native"
        assert payload["engine"]["name"] == "qwen"
        assert payload["audio"] == {
            "encoding": "s16le",
            "sample_rate_hz": 24_000,
            "channels": 1,
            "bits_per_sample": 16,
            "response_formats": ["pcm", "wav"],
            "pcm_media_type": "audio/pcm",
            "wav_media_type": "audio/wav",
        }

        _register(client)
        response = client.post(
            "/v1/audio/speech",
            headers={"X-Request-ID": "protocol-test-1"},
            json={"input": "audible", "voice": "mira"},
        )
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "protocol-test-1"
        assert response.headers["X-Audio-Encoding"] == "s16le"
        assert response.headers["X-Audio-Sample-Rate"] == "24000"
        assert response.headers["X-Audio-Channels"] == "1"
        assert response.headers["X-Audio-Bits-Per-Sample"] == "16"


def test_server_version_prefers_source_checkout_over_installed_metadata(monkeypatch):
    monkeypatch.setattr(
        qwen_server_module.importlib.metadata,
        "version",
        lambda _name: "0.7.4.dev7",
    )
    assert qwen_server_module._server_version() == "0.7.4.dev9"


def test_api_key_cors_and_lan_bind_defaults_are_restrictive(tmp_path, monkeypatch):
    parser = build_argument_parser()
    args = parser.parse_args(["--host", "0.0.0.0", "--allow-lan"])
    monkeypatch.delenv("REALTIMETTS_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        _validate_bind_security(args, _resolve_api_key(args, parser), parser)

    monkeypatch.setenv("REALTIMETTS_API_KEY", "from-environment")
    assert _resolve_api_key(args, parser) == "from-environment"
    _validate_bind_security(args, "from-environment", parser)

    with pytest.raises(ValueError, match="wildcard CORS"):
        _server(tmp_path, cors_origins=["*"])

    server = _server(
        tmp_path / "secured",
        api_key="secret",
        cors_origins=["http://localhost:3000"],
    )
    with TestClient(create_app(server)) as client:
        assert client.get("/health").status_code == 200
        unauthorized = client.get("/ready", headers={"X-Request-ID": "denied-1"})
        assert unauthorized.status_code == 401
        assert unauthorized.headers["X-Request-ID"] == "denied-1"
        assert client.get("/v1/capabilities").status_code == 401
        authorized = client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer secret"}
        )
        assert authorized.status_code == 200
        preflight = client.options(
            "/v1/capabilities",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "http://localhost:3000"
        assert "access-control-allow-origin" not in client.get(
            "/health", headers={"Origin": "http://evil.example"}
        ).headers


def test_shared_qwen_engine_rejects_multiple_active_workers_and_bounds_queue(tmp_path):
    engine = BlockingEngine()
    _persist_voice(tmp_path)
    server = _server(
        tmp_path,
        engine,
        max_active_requests=1,
        max_queued_requests=0,
        synthesis_timeout_seconds=2.0,
    )
    first = None
    try:
        request = _direct_request(server)
        first = server.start_synthesis(request, request_id="first")
        assert engine.entered.wait(1.0)
        with pytest.raises(ApiError) as raised:
            server.start_synthesis(request, request_id="second")
        assert raised.value.status == 429
        assert raised.value.error_type == "rate_limit_error"
    finally:
        engine.release.set()
        if first is not None and first.worker is not None:
            first.worker.join(1.0)
        server.shutdown()


def test_cancel_discards_buffered_and_late_pcm(tmp_path):
    engine = LateChunkEngine()
    _persist_voice(tmp_path)
    server = _server(tmp_path, engine, synthesis_timeout_seconds=2.0)
    state = server.start_synthesis(_direct_request(server), request_id="cancel-me")
    try:
        assert engine.first_chunk.wait(1.0)
        server.cancel(state)
        engine.release.set()
        assert state.worker is not None
        state.worker.join(1.0)
        assert list(server.iter_pcm(state)) == []
        assert state.cancelled.is_set()
    finally:
        engine.release.set()
        server.shutdown()


def test_synthesis_timeout_aborts_request_and_stops_engine(tmp_path):
    engine = BlockingEngine()
    _persist_voice(tmp_path)
    server = _server(tmp_path, engine, synthesis_timeout_seconds=0.05)
    state = server.start_synthesis(_direct_request(server), request_id="timeout-me")
    try:
        assert engine.entered.wait(1.0)
        assert _wait_for(lambda: state.timed_out, timeout=1.0)
        assert state.error_type == "timeout_error"
        assert state.error_status == 504
        assert state.cancelled.is_set()
        assert engine.stopped
    finally:
        engine.release.set()
        if state.worker is not None:
            state.worker.join(1.0)
        server.shutdown()


def test_shutdown_does_not_close_engine_until_workers_exit(tmp_path):
    engine = BlockingEngine()
    _persist_voice(tmp_path)
    server = _server(
        tmp_path,
        engine,
        synthesis_timeout_seconds=2.0,
        shutdown_timeout_seconds=0.01,
    )
    state = server.start_synthesis(_direct_request(server), request_id="shutdown-me")
    assert engine.entered.wait(1.0)
    server.shutdown()
    assert not engine.shutdown_called
    engine.release.set()
    assert state.worker is not None
    state.worker.join(1.0)
    server.shutdown()
    assert engine.shutdown_called


def test_websocket_terminal_events_carry_session_and_request_ids(tmp_path, monkeypatch):
    async def one_fragment(source, **_kwargs):
        async for chunk in source:
            if chunk.strip():
                yield chunk

    monkeypatch.setattr(qwen_server_module, "generate_sentences_async", one_fragment)
    _persist_voice(tmp_path)
    server = _server(tmp_path)
    with TestClient(create_app(server)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as websocket:
            websocket.send_json(
                {
                    "type": "config",
                    "session_id": "session-1",
                    "voice": "mira",
                    "language": "en",
                    "response_format": "pcm",
                }
            )
            websocket.send_json({"type": "text", "text": "audible"})
            websocket.send_json({"type": "end"})
            events = []
            while True:
                message = websocket.receive()
                if message.get("bytes") is not None:
                    continue
                event = json.loads(message["text"])
                events.append(event)
                if event["type"] == "done":
                    break
    assert events
    assert all(event["session_id"] == "session-1" for event in events)
    assert all("request_id" in event for event in events)
    assert events[-1]["type"] == "done"
    assert events[-1]["request_id"] == events[0]["request_id"]


def test_websocket_cancel_emits_cancelled_terminal_event(tmp_path, monkeypatch):
    async def one_fragment(source, **_kwargs):
        async for chunk in source:
            if chunk.strip():
                yield chunk

    monkeypatch.setattr(qwen_server_module, "generate_sentences_async", one_fragment)
    _persist_voice(tmp_path)
    engine = BlockingEngine()
    server = _server(tmp_path, engine, synthesis_timeout_seconds=2.0)
    try:
        with TestClient(create_app(server)) as client:
            with client.websocket_connect("/v1/audio/speech-stream") as websocket:
                websocket.send_json(
                    {
                        "type": "config",
                        "session_id": "session-cancel",
                        "voice": "mira",
                        "language": "en",
                        "response_format": "pcm",
                    }
                )
                websocket.send_json({"type": "text", "text": "audible"})
                fragment_ready = websocket.receive_json()
                assert fragment_ready["type"] == "fragment_ready"
                assert engine.entered.wait(1.0)
                websocket.send_json({"type": "cancel"})
                cancelled = websocket.receive_json()
                assert cancelled == {
                    "type": "cancelled",
                    "session_id": "session-cancel",
                    "request_id": fragment_ready["request_id"],
                }
                engine.release.set()
    finally:
        engine.release.set()
        server.shutdown()


def test_websocket_cancel_after_end_interrupts_active_audio(tmp_path, monkeypatch):
    async def one_fragment(source, **_kwargs):
        async for chunk in source:
            if chunk.strip():
                yield chunk

    monkeypatch.setattr(qwen_server_module, "generate_sentences_async", one_fragment)
    _persist_voice(tmp_path)
    engine = LateChunkEngine()
    server = _server(tmp_path, engine, synthesis_timeout_seconds=3.0)
    try:
        with TestClient(create_app(server)) as client:
            with client.websocket_connect("/v1/audio/speech-stream") as websocket:
                websocket.send_json(
                    {
                        "type": "config",
                        "session_id": "session-after-end-cancel",
                        "voice": "mira",
                        "language": "en",
                        "response_format": "pcm",
                    }
                )
                websocket.send_json({"type": "text", "text": "audible"})
                websocket.send_json({"type": "end"})

                fragment_ready = websocket.receive_json()
                assert fragment_ready["type"] == "fragment_ready"
                first_pcm = websocket.receive_json()
                assert first_pcm["type"] == "first_pcm_ready"
                assert websocket.receive_bytes() == engine.silent

                started = time.monotonic()
                websocket.send_json({"type": "cancel"})
                terminal = websocket.receive_json()
                elapsed = time.monotonic() - started

                assert terminal == {
                    "type": "cancelled",
                    "session_id": "session-after-end-cancel",
                    "request_id": fragment_ready["request_id"],
                }
                assert elapsed < 0.5
                assert engine.stopped
                engine.release.set()
    finally:
        engine.release.set()
        server.shutdown()


def test_websocket_pcm_sender_drops_chunk_when_cancel_wins_ready_race():
    class CancelOnReadyWebSocket:
        def __init__(self, cancelled):
            self.cancelled = cancelled
            self.events = []
            self.chunks = []

        async def send_json(self, event):
            self.events.append(event)
            self.cancelled.set()

        async def send_bytes(self, chunk):
            self.chunks.append(chunk)

    async def exercise():
        cancelled = asyncio.Event()
        websocket = CancelOnReadyWebSocket(cancelled)
        state = qwen_server_module.SynthesisState(
            "pcm",
            request_id="ready-race",
            output_queue_chunks=4,
            deadline=time.monotonic() + 2.0,
        )
        state.output.put(b"queued PCM")
        await qwen_server_module._send_stream_pcm(
            websocket,
            state,
            synthesis_started=time.monotonic(),
            first_fragment_timings={"first_fragment_chars": 10},
            session_id="ready-race-session",
            cancelled=cancelled,
        )
        return websocket

    websocket = asyncio.run(exercise())
    assert websocket.events[0]["type"] == "first_pcm_ready"
    assert websocket.chunks == []


def test_websocket_cancel_during_first_fragment_routing_starts_no_worker(
    tmp_path, monkeypatch
):
    async def one_fragment(source, **_kwargs):
        async for chunk in source:
            if chunk.strip():
                yield chunk

    class UncertainDetector:
        def detect(self, _text):
            return LanguageDetection(
                language="english",
                label="de",
                confidence=0.50,
                latency_ms=0.0,
                used_fallback=True,
                candidate_language="german",
                runner_up_label="en",
                runner_up_confidence=0.50,
                confidence_margin=0.0,
            )

    monkeypatch.setattr(qwen_server_module, "generate_sentences_async", one_fragment)
    _persist_voice(tmp_path)
    engine = BlockingEngine()
    router = QwenLanguageRouter(UncertainDetector())
    server = _server(
        tmp_path,
        engine,
        language_router=router,
        synthesis_timeout_seconds=3.0,
    )
    try:
        with TestClient(create_app(server)) as client:
            with client.websocket_connect("/v1/audio/speech-stream") as websocket:
                websocket.send_json(
                    {
                        "type": "config",
                        "session_id": "session-routing-cancel",
                        "voice": "mira",
                        "language": "Auto",
                        "response_format": "pcm",
                    }
                )
                websocket.send_json({"type": "text", "text": "short fragment"})
                fragment_ready = websocket.receive_json()
                assert fragment_ready["type"] == "fragment_ready"

                websocket.send_json({"type": "cancel"})
                terminal = websocket.receive_json()

                assert terminal == {
                    "type": "cancelled",
                    "session_id": "session-routing-cancel",
                    "request_id": fragment_ready["request_id"],
                }
                assert not engine.entered.is_set()
                assert client.get("/health").json()["requests_total"] == 0
    finally:
        engine.release.set()
        server.shutdown()


def test_models_voice_registry_persistence_and_delete(tmp_path):
    first_engine = FakeEngine()
    first = _server(tmp_path, first_engine, alias="production-qwen")
    with TestClient(create_app(first)) as client:
        assert client.get("/v1/models").json() == {
            "object": "list",
            "data": [
                {"id": "production-qwen", "object": "model", "owned_by": "local"}
            ],
        }
        assert client.get("/v1/audio/voices").json() == {"voices": []}
        _register(client)
        assert client.get("/v1/audio/voices").json() == {
            "voices": [{"name": "mira", "kind": "registered"}]
        }
    assert first_engine.shutdown_called

    second = _server(tmp_path)
    with TestClient(create_app(second)) as client:
        assert client.get("/v1/audio/voices").json() == {
            "voices": [{"name": "mira", "kind": "registered"}]
        }
        assert client.delete("/v1/audio/voices/mira").json() == {"status": "deleted"}
        missing = client.delete("/v1/audio/voices/mira")
        assert missing.status_code == 404
        assert missing.json()["error"]["type"] == "not_found_error"


def test_startup_warmup_prepares_persistent_voice_without_queue_leaks(tmp_path):
    _persist_voice(tmp_path)
    engine = FakeEngine()
    response_queue = engine.queue
    server = _server(tmp_path, engine, startup_warmup_voice="mira")

    with TestClient(create_app(server)) as client:
        assert engine.warmup_calls == 1
        assert engine.current_voice.name == "mira"
        assert engine.queue is response_queue
        assert response_queue.empty()
        health = client.get("/health").json()
        assert health["requests_total"] == 0
        server.startup()
        assert engine.warmup_calls == 1
        response = client.post(
            "/v1/audio/speech",
            json={"input": "audible", "voice": "mira", "response_format": "pcm"},
        )
        assert response.status_code == 200
        assert response.content == engine.silent + engine.audible


def test_startup_warmup_fails_before_ready_when_voice_is_missing(tmp_path):
    engine = FakeEngine()
    server = _server(tmp_path, engine, startup_warmup_voice="missing")

    try:
        with TestClient(create_app(server)):
            raise AssertionError("application unexpectedly became ready")
    except RuntimeError as exc:
        assert "startup warmup voice 'missing' is not registered" in str(exc)
    assert engine.shutdown_called


def test_startup_warmup_cli_is_opt_in():
    parser = build_argument_parser()
    defaults = parser.parse_args([])
    configured = parser.parse_args(
        [
            "--startup-warmup-voice",
            "mira",
            "--startup-warmup-text",
            "Short warmup.",
            "--startup-warmup-tokens",
            "8",
        ]
    )

    assert defaults.startup_warmup_voice is None
    assert defaults.startup_warmup_tokens == 32
    assert defaults.clamp_fp16 is True
    assert defaults.trim_silence is True
    assert defaults.silence_threshold == 0.005
    assert defaults.trim_pre_roll_ms == 15.0
    assert defaults.trim_fade_in_ms == 20.0
    assert defaults.startup_buffer_ms == 160.0
    assert configured.startup_warmup_voice == "mira"
    assert configured.startup_warmup_text == "Short warmup."
    assert configured.startup_warmup_tokens == 8
    assert parser.parse_args(["--no-clamp-fp16"]).clamp_fp16 is False
    assert parser.parse_args(["--no-trim-silence"]).trim_silence is False


def test_pcm_waits_for_audible_audio_and_preserves_leading_silence(tmp_path):
    engine = FakeEngine()
    server = _server(tmp_path, engine)
    with TestClient(create_app(server)) as client:
        _register(client)
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "audible",
                "voice": "mira",
                "response_format": "pcm",
                "seed": 42,
                "temperature": 0.7,
                "max_new_tokens": 128,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/pcm")
        assert response.content == engine.silent + engine.audible
        assert engine.parameters["seed"] == 42
        assert engine.parameters["temperature"] == 0.7
        assert engine.parameters["max_new_tokens"] == 128
        health = client.get("/health").json()
        assert health == {
            "status": "ok",
            "active_requests": 0,
            "requests_total": 1,
            "synthesis_failures_total": 0,
            "unusable_outputs_total": 0,
            "last_progress_age_ms": 0,
            "stalled": False,
        }


def test_silent_pcm_is_a_real_502_and_does_not_mark_health_down(tmp_path):
    server = _server(tmp_path)
    with TestClient(create_app(server)) as client:
        _register(client)
        response = client.post(
            "/v1/audio/speech",
            json={"input": "silent", "voice": "mira", "response_format": "pcm"},
        )
        assert response.status_code == 502
        assert response.json() == {
            "error": {
                "message": "synthesis produced no audible audio",
                "type": "output_error",
            }
        }
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["unusable_outputs_total"] == 1
        assert health.json()["synthesis_failures_total"] == 0


def test_empty_native_output_is_classified_as_unusable_not_backend_failure(tmp_path):
    server = _server(tmp_path)
    with TestClient(create_app(server)) as client:
        _register(client)
        response = client.post(
            "/v1/audio/speech",
            json={"input": "empty", "voice": "mira"},
        )
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "output_error"
        health = client.get("/health").json()
        assert health["unusable_outputs_total"] == 1
        assert health["synthesis_failures_total"] == 0


def test_backend_failure_returns_server_error_and_updates_metric(tmp_path):
    server = _server(tmp_path)
    with TestClient(create_app(server)) as client:
        _register(client)
        response = client.post(
            "/v1/audio/speech",
            json={"input": "failure", "voice": "mira"},
        )
        assert response.status_code == 502
        assert response.json() == {
            "error": {"message": "native synthesis failed", "type": "server_error"}
        }
        health = client.get("/health").json()
        assert health["synthesis_failures_total"] == 1
        assert health["unusable_outputs_total"] == 0


def test_wav_is_complete_valid_24khz_mono_pcm16(tmp_path):
    engine = FakeEngine()
    server = _server(tmp_path, engine)
    with TestClient(create_app(server)) as client:
        _register(client)
        response = client.post(
            "/v1/audio/speech",
            json={"input": "audible", "voice": "mira", "response_format": "wav"},
        )
        assert response.status_code == 200
        assert response.headers["X-Audio-Sample-Rate"] == "24000"
        assert response.headers["X-Audio-Channels"] == "1"
        with wave.open(io.BytesIO(response.content), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == 24_000
            assert wav.readframes(wav.getnframes()) == engine.silent + engine.audible


def test_registration_keeps_icl_text_and_request_instructions(tmp_path):
    engine = FakeEngine()
    server = _server(tmp_path, engine)
    with TestClient(create_app(server)) as client:
        _register(client, ref_text="the exact reference words")
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "audible",
                "voice": "mira",
                "instructions": "sound excited",
            },
        )
        assert response.status_code == 200
        assert engine.current_voice.ref_text == "the exact reference words"
        assert engine.current_voice.instruct == "sound excited"


def test_request_language_reaches_native_voice_and_defaults_to_auto(tmp_path):
    engine = FakeEngine()
    server = _server(tmp_path, engine)
    with TestClient(create_app(server)) as client:
        _register(client)
        response = client.post(
            "/v1/audio/speech",
            json={"input": "Guten Tag.", "voice": "mira", "language": "German"},
        )
        assert response.status_code == 200
        assert engine.current_voice.language == "german"

        response = client.post(
            "/v1/audio/speech",
            json={"input": "Hello.", "voice": "mira"},
        )
        assert response.status_code == 200
        assert engine.current_voice.language == "auto"


def test_auto_language_routes_native_language_and_matching_reference(tmp_path):
    class GermanDetector:
        def warmup(self, _text):
            return self.detect("This is the detector warmup sentence.")

        def detect(self, _text):
            return LanguageDetection("german", "de", 0.99, 0.03)

    _persist_voice(tmp_path, name="mira_v5_spark", ref_text="English reference")
    _persist_voice(tmp_path, name="mira_v5_spark_de", ref_text="Deutsche Referenz")
    engine = FakeEngine()
    router = QwenLanguageRouter(
        GermanDetector(),
        voice_routes={"mira_v5_spark": {"de": "mira_v5_spark_de"}},
    )
    server = _server(tmp_path, engine, language_router=router)

    with TestClient(create_app(server)) as client:
        response = client.post(
            "/v1/audio/speech",
            json={
                "input": "Das ist ein deutscher Satz.",
                "voice": "mira_v5_spark",
                "language": "Auto",
            },
        )

    assert response.status_code == 200
    assert engine.current_voice.name == "mira_v5_spark_de"
    assert engine.current_voice.language == "german"
    assert engine.current_voice.ref_text == "Deutsche Referenz"


def test_stream_lookahead_reclassifies_then_locks_german_for_the_response(
    tmp_path, monkeypatch
):
    async def fragment_stream(source, **_kwargs):
        pending = ""
        first = True
        async for chunk in source:
            pending += chunk
            if first and "," in pending:
                fragment, pending = pending.rsplit(",", 1)
                first = False
                yield fragment + ","
        if pending.strip():
            yield pending.strip()

    monkeypatch.setattr(
        qwen_server_module, "generate_sentences_async", fragment_stream
    )

    class PrefixDetector:
        def __init__(self):
            self.calls = []

        def warmup(self, _text):
            return LanguageDetection("english", "en", 0.99, 0.01)

        def detect(self, text):
            self.calls.append(text)
            if " ich" in text:
                return LanguageDetection(
                    "german",
                    "de",
                    0.9979,
                    0.01,
                    candidate_language="german",
                    runner_up_label="en",
                    runner_up_confidence=0.0004,
                    confidence_margin=0.9975,
                )
            return LanguageDetection(
                "english",
                "de",
                0.4669775068759918,
                0.01,
                used_fallback=True,
                candidate_language="german",
                runner_up_label="tr",
                runner_up_confidence=0.281007319688797,
                confidence_margin=0.18597018718719482,
            )

    class RecordingEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.synthesis_calls = []

        def synthesize(self, text):
            self.synthesis_calls.append(
                (text, self.current_voice.name, self.current_voice.language)
            )
            return super().synthesize(text)

    for name, ref_text in (
        ("mira", "Base reference"),
        ("mira_en", "English reference"),
        ("mira_de", "Deutsche Referenz"),
    ):
        _persist_voice(tmp_path, name=name, ref_text=ref_text)
    detector = PrefixDetector()
    engine = RecordingEngine()
    router = QwenLanguageRouter(
        detector,
        voice_routes={"mira": {"en": "mira_en", "de": "mira_de"}},
    )
    server = _server(tmp_path, engine, language_router=router)

    def drain(websocket):
        audio = bytearray()
        while True:
            message = websocket.receive()
            if message.get("bytes") is not None:
                audio.extend(message["bytes"])
                continue
            event = json.loads(message["text"])
            if event["type"] == "done":
                return audio

    with TestClient(create_app(server)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as websocket:
            websocket.send_json(
                {
                    "type": "config",
                    "voice": "mira",
                    "language": "Auto",
                    "response_format": "pcm",
                }
            )
            websocket.send_json({"type": "text", "text": "Ja, nat\u00fcrlich,"})
            first_event = websocket.receive_json()
            assert first_event["type"] == "fragment_ready"

            websocket.send_json({"type": "text", "text": " ich kann helfen."})
            websocket.send_json({"type": "end"})
            audio = drain(websocket)

    assert audio
    assert detector.calls[0] == "Ja, nat\u00fcrlich,"
    assert any(" ich" in text for text in detector.calls[1:])
    assert engine.synthesis_calls
    assert engine.synthesis_calls[0][0] == "Ja, nat\u00fcrlich,"
    assert all(
        voice == "mira_de" and language == "german"
        for _text, voice, language in engine.synthesis_calls
    )

    detector.calls.clear()
    engine.synthesis_calls.clear()
    server = _server(tmp_path, engine, language_router=router)
    with TestClient(create_app(server)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as websocket:
            websocket.send_json(
                {
                    "type": "config",
                    "voice": "mira",
                    "language": "Auto",
                    "response_format": "pcm",
                }
            )
            websocket.send_json({"type": "text", "text": "Ja, nat\u00fcrlich,"})
            assert websocket.receive_json()["type"] == "fragment_ready"
            websocket.send_json({"type": "end"})
            assert drain(websocket)

    assert detector.calls
    assert all(text == "Ja, nat\u00fcrlich," for text in detector.calls)
    assert engine.synthesis_calls == [("Ja, nat\u00fcrlich,", "mira_de", "german")]


def test_request_language_rejects_invalid_values(tmp_path):
    server = _server(tmp_path)
    with TestClient(create_app(server)) as client:
        _register(client)
        for language in ("", "   ", 7, None):
            response = client.post(
                "/v1/audio/speech",
                json={"input": "Hello.", "voice": "mira", "language": language},
            )
            assert response.status_code == 400
            assert response.json()["error"]["type"] == "invalid_request_error"


def test_invalid_requests_use_native_server_error_envelope(tmp_path):
    server = _server(tmp_path)
    with TestClient(create_app(server)) as client:
        invalid_json = client.post(
            "/v1/audio/voices",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert invalid_json.status_code == 400
        assert invalid_json.json()["error"]["type"] == "invalid_request_error"

        half_latents = client.post(
            "/v1/audio/voices",
            json={"name": "bad", "spk_b64": base64.b64encode(b"x").decode("ascii")},
        )
        assert half_latents.status_code == 400

        unknown = client.post(
            "/v1/audio/speech",
            json={"input": "hello", "voice": "missing"},
        )
        assert unknown.status_code == 400
        assert unknown.json()["error"]["type"] == "invalid_request_error"


def test_health_degrades_only_after_active_request_stalls():
    now = [100.0]
    metrics = RequestMetrics(stall_timeout_seconds=30.0, clock=lambda: now[0])
    initial, status = metrics.snapshot()
    assert status == 200
    assert initial["last_progress_age_ms"] == 0
    metrics.started()
    now[0] += 29.9
    healthy, status = metrics.snapshot()
    assert status == 200
    assert not healthy["stalled"]
    now[0] += 0.2
    stalled, status = metrics.snapshot()
    assert status == 503
    assert stalled["status"] == "degraded"
    assert stalled["stalled"]
    assert 30_099 <= stalled["last_progress_age_ms"] <= 30_100
    metrics.finished()
    recovered, status = metrics.snapshot()
    assert status == 200
    assert recovered["last_progress_age_ms"] == 0
    assert not recovered["stalled"]


def test_health_endpoint_returns_503_for_the_watchdog_only_while_stalled(tmp_path):
    now = [10.0]
    server = _server(
        tmp_path,
        stall_timeout_seconds=30.0,
        clock=lambda: now[0],
    )
    with TestClient(create_app(server)) as client:
        server.metrics.started()
        now[0] += 30.0
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"
        assert response.json()["stalled"] is True
        server.metrics.progress()
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        server.metrics.finished()
