"""OpenAI SDK/HTTP contract tests; all requests and audio devices are local fakes."""

import json
import threading
import wave

import httpx
import numpy as np
import pytest

openai = pytest.importorskip("openai")

from RealtimeTTS import TextToAudioStream
from RealtimeTTS._audio_backend import pyaudio
from RealtimeTTS.engines import openai_engine
from tests.test_alignment_decoder import _mp3_fixture, _decode_in_chunks
from tests.test_text_to_stream_alignment import _FakePyAudio
import RealtimeTTS.stream_player as player_module


class AudioBody(httpx.SyncByteStream):
    def __init__(self, chunks, *, gate_at=None, error=None):
        self.chunks = chunks
        self.gate_at = gate_at
        self.error = error
        self.waiting = threading.Event()
        self.release = threading.Event()
        self.closed = False

    def __iter__(self):
        for index, chunk in enumerate(self.chunks):
            if index == self.gate_at:
                self.waiting.set()
                if not self.release.wait(5):
                    raise AssertionError("audio consumer did not release HTTP body")
            yield chunk
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


@pytest.fixture
def engine_factory(monkeypatch):
    clients = []
    requests = []
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    def make(body, **options):
        def handler(request):
            requests.append(request)
            response_body = body() if callable(body) else body
            return httpx.Response(200, stream=response_body)

        def client(**kwargs):
            result = openai.OpenAI(
                **kwargs,
                http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                max_retries=0,
            )
            clients.append(result)
            return result

        monkeypatch.setattr(openai_engine, "OpenAI", client)
        return openai_engine.OpenAIEngine(api_key="test-only-key", **options), requests

    yield make
    for client in clients:
        client.close()


def _worker(action):
    errors = []
    results = []

    def run():
        try:
            results.append(action())
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    return worker, results, errors


@pytest.mark.parametrize("response_format", ["pcm", "mp3"])
def test_audio_reaches_queue_before_http_body_finishes(engine_factory, response_format):
    body = AudioBody([b"\x01\x00" * 100, b"\x02\x00" * 100], gate_at=1)
    engine, _ = engine_factory(body, response_format=response_format)
    worker, results, errors = _worker(lambda: engine.synthesize("Hello."))
    try:
        assert body.waiting.wait(2)
        assert engine.queue.get(timeout=1) == body.chunks[0]
        assert worker.is_alive()
        assert not body.closed
    finally:
        body.release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert not errors
    assert results == [True]
    assert engine.queue.get_nowait() == body.chunks[1]
    assert body.closed


@pytest.mark.parametrize(
    "environment,api_url,expected",
    [
        (None, None, "https://api.openai.com/v1/audio/speech"),
        ("https://env.example/v1", None, "https://env.example/v1/audio/speech"),
        ("https://env.example/v1", "https://chosen.example/tts/v1",
         "https://chosen.example/tts/v1/audio/speech"),
    ],
)
def test_endpoint_and_request_options(engine_factory, monkeypatch, environment, api_url, expected):
    if environment:
        monkeypatch.setenv("OPENAI_BASE_URL", environment)
    body = AudioBody([b"\x01\x00"])
    engine, requests = engine_factory(
        body, api_url=api_url, model="custom-tts", voice="custom-voice",
        instructions="Speak softly.", speed=1.25, response_format="PCM", timeout=7.5,
    )
    assert engine.synthesize("Endpoint test.")
    request = requests[-1]
    assert str(request.url) == expected
    assert json.loads(request.content) == {
        "model": "custom-tts", "voice": "custom-voice", "input": "Endpoint test.",
        "instructions": "Speak softly.", "speed": 1.25, "response_format": "pcm",
    }
    assert request.extensions["timeout"]["read"] == 7.5
    assert request.headers["authorization"] == "Bearer test-only-key"
    assert engine.get_stream_info() == (pyaudio.paInt16, 1, 24000)


def test_endpoints_are_per_engine(engine_factory):
    first, requests = engine_factory(
        AudioBody([b"\0\0"]), api_url="https://first.example/v1", response_format="pcm"
    )
    second, _ = engine_factory(
        AudioBody([b"\0\0"]), api_url="https://second.example/v1", response_format="pcm"
    )
    assert first.synthesize("First.")
    assert second.synthesize("Second.")
    assert [request.url.host for request in requests] == ["first.example", "second.example"]


def test_pcm_buffers_split_samples_without_loss(engine_factory):
    body = AudioBody([b"\x01", b"\x00\x02", b"\x00\x03\x00"])
    engine, _ = engine_factory(body, response_format="pcm")
    assert engine.synthesize("Split samples.")
    chunks = []
    while not engine.queue.empty():
        chunks.append(engine.queue.get_nowait())
    assert chunks and all(len(chunk) % 2 == 0 for chunk in chunks)
    assert b"".join(chunks) == b"\x01\x00\x02\x00\x03\x00"
    assert body.closed


def test_incomplete_pcm_sample_is_reported_and_response_closed(engine_factory):
    body = AudioBody([b"\x01"])
    engine, _ = engine_factory(body, response_format="pcm")
    with pytest.raises(ValueError, match="incomplete PCM sample"):
        engine.synthesize("Truncated.")
    assert body.closed
    assert engine.queue.empty()


def test_read_failure_closes_response_without_replaying_audio(engine_factory):
    body = AudioBody([b"\0\0"], error=httpx.ReadError("interrupted"))
    engine, requests = engine_factory(body, response_format="pcm")
    with pytest.raises(httpx.ReadError, match="interrupted"):
        engine.synthesize("Interrupted.")
    assert body.closed
    assert len(requests) == 1
    assert engine.queue.get_nowait() == b"\0\0"
    assert engine.queue.empty()


def test_stop_discards_late_chunks_and_closes_response(engine_factory):
    body = AudioBody([b"\x01\x00", b"\x02\x00"], gate_at=1)
    engine, _ = engine_factory(body, response_format="pcm")
    worker, results, errors = _worker(lambda: engine.synthesize("Stop."))
    try:
        assert body.waiting.wait(2)
        assert engine.queue.get(timeout=1) == body.chunks[0]
        engine.stop()
    finally:
        body.release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert not errors
    assert results == [False]
    assert engine.queue.empty()
    assert body.closed


@pytest.mark.parametrize("silence_s", [0, 0.05])
@pytest.mark.parametrize("response_format", ["pcm", "mp3"])
def test_stream_plays_before_eof_and_exports_continuous_audio(
    engine_factory, monkeypatch, tmp_path, response_format, silence_s
):
    class FakeAudio(_FakePyAudio):
        def is_format_supported(self, rate, **kwargs):
            return rate == 24000

    monkeypatch.setattr(player_module.pyaudio, "PyAudio", FakeAudio)
    if response_format == "mp3":
        payload = _mp3_fixture()
        expected = np.int16(_decode_in_chunks(payload, [len(payload)]) * 32767).tobytes()
    else:
        payload = np.arange(24000, dtype="<i2").tobytes()
        expected = payload
    chunks = [payload[i:i + 73] for i in range(0, len(payload), 73)]
    expected += b"\0\0" * round(24000 * silence_s)
    bodies = []

    def next_body():
        body = AudioBody(chunks, gate_at=len(chunks) // 2 if not bodies else None)
        bodies.append(body)
        return body

    engine, requests = engine_factory(next_body, response_format=response_format)
    stream = TextToAudioStream(engine, muted=True, tokenizer="rule-based")
    received = []
    audible = threading.Event()

    def on_chunk(chunk):
        received.append(chunk)
        audible.set()

    output = tmp_path / "stream.wav"
    worker, _, errors = _worker(
        lambda: stream.feed("Hello.").play(
            on_audio_chunk=on_chunk, output_wavfile=str(output),
            sentence_silence_duration=silence_s, default_silence_duration=0,
        )
    )
    try:
        assert audible.wait(3), "no playable audio before the response completed"
        assert bodies[0].waiting.wait(2)
        assert worker.is_alive()
    finally:
        if bodies:
            bodies[0].release.set()
        worker.join(5)
    try:
        assert not worker.is_alive()
        assert not errors
        assert not stream.error_flag
        assert b"".join(received) == expected
        with wave.open(str(output), "rb") as wav:
            assert wav.getframerate() == 24000
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.readframes(wav.getnframes()) == expected
        # A second request starts with a fresh decoder and no previous response tail.
        received.clear()
        stream.feed("Again.").play(
            on_audio_chunk=received.append,
            sentence_silence_duration=silence_s, default_silence_duration=0,
        )
        assert not stream.error_flag
        assert b"".join(received) == expected
        assert len(requests) == 2
        assert all(body.closed for body in bodies)
    finally:
        stream.stop()


@pytest.mark.parametrize("stop_between_streams", [False, True])
def test_mp3_engine_reuse_preserves_legacy_alignment_capture(
    engine_factory, monkeypatch, stop_between_streams
):
    from RealtimeTTS.alignment import FakeWordAligner, WordAlignment

    class FakeAudio(_FakePyAudio):
        def is_format_supported(self, rate, **kwargs):
            return rate == 24000

    monkeypatch.setattr(player_module.pyaudio, "PyAudio", FakeAudio)
    payload = _mp3_fixture()
    expected = np.int16(_decode_in_chunks(payload, [len(payload)]) * 32767).tobytes()
    engine, _ = engine_factory(
        lambda: AudioBody([payload[i:i + 73] for i in range(0, len(payload), 73)])
    )
    streams = []
    try:
        for align_words in (False, True, False):
            aligner = FakeWordAligner([WordAlignment("Hello", 0, 0.4)])
            stream = TextToAudioStream(
                engine, muted=True, tokenizer="rule-based",
                align_words=align_words, word_aligner=aligner, alignment_blocking=True,
            )
            streams.append(stream)
            chunks = []
            stream.feed("Hello.").play(
                on_audio_chunk=chunks.append,
                sentence_silence_duration=0, default_silence_duration=0,
            )
            assert not stream.error_flag
            assert b"".join(chunks) == expected
            if align_words:
                assert aligner.calls == [("Hello.", len(expected) // 2, 24000)]
            else:
                assert aligner.calls == []
            if stop_between_streams:
                stream.stop()
    finally:
        for stream in streams:
            stream.stop()
