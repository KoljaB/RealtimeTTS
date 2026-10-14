import threading
import time

import pytest

from RealtimeTTS._audio_formats import INT16
from RealtimeTTS.engines.qwen_server_engine import QwenServerEngine, QwenServerVoice


_HEADERS = {
    "Content-Type": "audio/pcm",
    "X-Audio-Encoding": "s16le",
    "X-Audio-Sample-Rate": "24000",
    "X-Audio-Channels": "1",
    "X-Audio-Bits-Per-Sample": "16",
}


class FakeResponse:
    def __init__(self, chunks=(), headers=None, error=None, payload=None, wait_event=None):
        self.chunks = list(chunks)
        self.headers = dict(headers or _HEADERS)
        self.error = error
        self.payload = payload
        self.wait_event = wait_event
        self.closed = False

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def iter_content(self, chunk_size=None):
        assert chunk_size is None
        for chunk in self.chunks:
            if self.wait_event is not None:
                self.wait_event.wait(2)
            yield chunk

    def json(self):
        return self.payload

    def close(self):
        self.closed = True


class BlockingBytes:
    def __init__(self, payload, started, release):
        self.payload = payload
        self.started = started
        self.release = release

    def __bytes__(self):
        self.started.set()
        self.release.wait(2)
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def make_engine(monkeypatch, responses, **kwargs):
    session = FakeSession(responses)
    monkeypatch.setattr(
        "RealtimeTTS.engines.qwen_server_engine.requests.Session",
        lambda: session,
    )
    return QwenServerEngine(**kwargs), session


def test_streams_pcm_payload_auth_and_frame_splits(monkeypatch):
    pcm = b"\x01\x00\x02\x00\x03\x00"
    response = FakeResponse([pcm[:1], pcm[1:4], pcm[4:]])
    engine, session = make_engine(
        monkeypatch,
        [response],
        api_key="secret-token",
        voice="alex",
        language="english",
        seed=7,
        clone_mode="speaker_only",
        max_new_tokens=321,
        temperature=0.5,
    )

    assert engine.get_stream_info() == (INT16, 1, 24000)
    assert engine.synthesize("Hello") is True
    assert engine.queue.get_nowait() + engine.queue.get_nowait() == pcm
    request = session.calls[0]
    assert request[:2] == ("POST", "http://127.0.0.1:18085/v1/audio/speech")
    assert request[2]["stream"] is True
    assert request[2]["json"] == {
        "input": "Hello",
        "voice": "alex",
        "language": "english",
        "clone_mode": "speaker_only",
        "seed": 7,
        "max_new_tokens": 321,
        "temperature": 0.5,
        "response_format": "pcm",
    }
    assert request[2]["headers"]["Authorization"] == "Bearer secret-token"
    assert "secret-token" not in repr(request[2]["json"])
    assert engine.last_chunk_count == 2
    assert engine.last_first_audio_time is not None
    assert engine.last_first_audio_ms is not None
    assert engine.last_wall_time is not None


def test_get_voices_and_qwen_voice_name_without_upload(monkeypatch):
    response = FakeResponse(payload={"voices": [{"name": "alex", "kind": "registered"}]})
    engine, session = make_engine(monkeypatch, [response])
    assert engine.get_voices() == [QwenServerVoice("alex", "registered")]
    engine.set_voice(type("QwenLike", (), {"name": "mira"})())
    engine.set_voice_parameters(language="german", seed=3, max_new_tokens=9, temperature=0.2)
    assert engine.voice == "mira"
    assert session.calls[0][0] == "GET"
    assert session.calls[0][1].endswith("/v1/audio/voices")


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse([], headers=_HEADERS),
        FakeResponse([b"\x01"], headers=_HEADERS),
        FakeResponse([b"\x00\x00"], headers={**_HEADERS, "X-Audio-Sample-Rate": "16000"}),
        FakeResponse([b"\x00\x00"], headers={**_HEADERS, "X-Audio-Encoding": "f32le"}),
        FakeResponse([b"\x00\x00"], headers={**_HEADERS, "Content-Type": "text/plain"}),
    ],
)
def test_rejects_empty_truncated_or_wrong_contract(monkeypatch, response):
    engine, _session = make_engine(monkeypatch, [response])
    assert engine.synthesize("Hello") is False
    assert engine.last_error is not None
    assert engine.queue.empty()


def test_http_error_does_not_leak_auth(monkeypatch):
    response = FakeResponse(error=RuntimeError("server returned 502"))
    engine, session = make_engine(monkeypatch, [response], api_key="private-key")
    assert engine.synthesize("Hello") is False
    assert "private-key" not in repr(engine.last_error)
    assert "private-key" not in repr(session.calls[0][2]["json"])


def test_stop_closes_active_response_and_cooperates(monkeypatch):
    gate = threading.Event()
    response = FakeResponse([b"\x01\x00"], wait_event=gate)
    engine, _session = make_engine(monkeypatch, [response])
    worker = threading.Thread(target=engine.synthesize, args=("Hello",))
    worker.start()
    for _ in range(100):
        if engine._active_response is response:
            break
        time.sleep(0.005)
    engine.stop()
    gate.set()
    worker.join(2)
    assert not worker.is_alive()
    assert response.closed
    assert engine.last_error is None


def test_stop_fences_chunk_after_stream_check(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    response = FakeResponse([BlockingBytes(b"\x01\x00", started, release)])
    engine, _session = make_engine(monkeypatch, [response])
    worker = threading.Thread(target=engine.synthesize, args=("Hello",))
    worker.start()
    assert started.wait(2)
    engine.stop()
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert engine.queue.empty()
    assert engine.last_error is None


def test_shutdown_closes_session_and_active_response(monkeypatch):
    gate = threading.Event()
    response = FakeResponse([b"\x01\x00"], wait_event=gate)
    engine, session = make_engine(monkeypatch, [response])
    worker = threading.Thread(target=engine.synthesize, args=("Hello",))
    worker.start()
    for _ in range(100):
        if engine._active_response is response:
            break
        time.sleep(0.005)
    engine.shutdown()
    gate.set()
    worker.join(2)
    assert not worker.is_alive()
    assert response.closed
    assert session.closed
