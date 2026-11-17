"""
Unit tests for GmicloudEngine.

Run with: pytest tests/test_gmicloud_engine.py -v

Collection is light: gmicloud_engine imports only numpy and requests (PyAudio
is imported lazily inside get_stream_info), so no optional wheels are needed.
"""

import io
import queue as _queue
import struct
import wave

import pytest

import RealtimeTTS.engines.gmicloud_engine as engine_mod
from RealtimeTTS.engines.gmicloud_engine import (
    GmicloudEngine,
    GmicloudVoice,
    _extract_audio_url,
    _parse_wav,
    _resample_s16,
    _to_mono_s16,
    _TTSOptions,
    build_payload,
    model_family,
)


def _opts(**kw):
    base = {
        "api_key": "k",
        "base_url": "https://console.gmicloud.ai",
        "model": "inworld-tts-2",
        "voice": None,
        "audio_format": "WAV",
        "speed": 1.0,
        "temperature": None,
        "timestamp_type": None,
    }
    base.update(kw)
    return _TTSOptions(**base)


def _wav(pcm: bytes, rate: int, channels: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


# -- family mapping ---------------------------------------------------------

def test_model_family_mapping():
    assert model_family("inworld-tts-2") == "inworld"
    assert model_family("minimax-tts-speech-2.8-turbo") == "minimax"
    assert model_family("Chatterbox-tts") == "chatterbox"
    assert model_family("some-future-model") == "inworld"
    assert model_family("") == "inworld"


# -- payload building -------------------------------------------------------

def test_inworld_payload_shape():
    p = build_payload("hello", _opts(voice="Dennis", speed=1.2, temperature=0.5))
    assert p == {
        "text": "hello",
        "audio_encoding": "WAV",
        "speaking_rate": 1.2,
        "voice_id": "Dennis",
        "temperature": 0.5,
    }


def test_inworld_default_voice():
    p = build_payload("hello", _opts())
    assert p["voice_id"] == "Dennis"
    assert "temperature" not in p


def test_minimax_payload_shape():
    p = build_payload("hello", _opts(model="minimax-tts-speech-2.8-turbo", speed=1.5))
    assert p == {
        "text": "hello",
        "speed": "1.5",
        "format": "wav",
        "voice_id": "English_expressive_narrator",
    }


def test_chatterbox_payload_shape():
    p = build_payload("hello", _opts(model="Chatterbox-tts", voice="uuid-123"))
    assert p == {"data": "hello", "voice_uuid": "uuid-123", "output_format": "wav"}


def test_chatterbox_without_voice_raises():
    with pytest.raises(ValueError, match="voice_uuid"):
        build_payload("hello", _opts(model="Chatterbox-tts"))


def test_extra_payload_overrides():
    p = build_payload("hello", _opts(extra_payload={"voice_id": "Override"}))
    assert p["voice_id"] == "Override"


# -- URL extraction ---------------------------------------------------------

def test_extract_audio_url_shapes():
    assert _extract_audio_url({"outcome": {"audio_url": "u"}}) == "u"
    assert _extract_audio_url({"outcome": {"media_urls": [{"url": "v"}]}}) == "v"
    assert _extract_audio_url({"outcome": {"medias": [{"url": "w"}]}}) == "w"
    assert _extract_audio_url({"outcome": {}}) is None
    assert _extract_audio_url({}) is None


# -- WAV handling and normalization -----------------------------------------

def test_parse_wav_roundtrip_and_garbage():
    pcm = b"\x01\x00" * 100
    parsed = _parse_wav(_wav(pcm, 48000, 1))
    assert parsed == (pcm, 48000, 1, 2)
    assert _parse_wav(b"not a wav") is None


def test_to_mono_s16_downmixes_by_averaging():
    stereo = struct.pack("<4h", 1000, 3000, 2000, 4000)
    assert struct.unpack("<2h", _to_mono_s16(stereo, 2)) == (2000, 3000)


def test_to_mono_s16_passthrough():
    mono = b"\x01\x00" * 50
    assert _to_mono_s16(mono, 1) == mono


def test_resample_changes_length_and_identity():
    pcm = b"\x01\x00" * 32000
    out = _resample_s16(pcm, 32000, 48000)
    assert 47900 <= len(out) // 2 <= 48100
    assert _resample_s16(pcm, 32000, 32000) == pcm


# -- engine construction ----------------------------------------------------

def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GMI_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("GMI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        GmicloudEngine()


def test_chatterbox_without_voice_raises_on_init():
    with pytest.raises(ValueError, match="voice_uuid"):
        GmicloudEngine(api_key="k", model="Chatterbox-tts")


def test_defaults_and_voices():
    e = GmicloudEngine(api_key="k")
    assert e.model == "inworld-tts-2"
    assert e.engine_name == "gmicloud"
    voices = e.get_voices()
    assert {v.name for v in voices} == {"Dennis", "English_expressive_narrator"}
    assert all(isinstance(v, GmicloudVoice) for v in voices)


def test_set_voice_and_parameters():
    e = GmicloudEngine(api_key="k")
    e.set_voice("Custom-Voice")
    assert e.voice == "Custom-Voice"
    e.set_voice(GmicloudVoice("Dennis"))
    assert e.voice == "Dennis"
    e.set_voice_parameters(speed=2.0, model="minimax-tts-speech-2.8-turbo", temperature=0.7)
    assert (e.speed, e.model, e.temperature) == (2.0, "minimax-tts-speech-2.8-turbo", 0.7)


def test_get_stream_info_declares_mono_16bit():
    pytest.importorskip("pyaudio")
    import pyaudio

    e = GmicloudEngine(api_key="k")
    fmt, channels, rate = e.get_stream_info()
    assert (fmt, channels, rate) == (pyaudio.paInt16, 1, 48000)


# -- request flow (fake HTTP) -----------------------------------------------

class FakeResp:
    def __init__(self, status_code=200, json_data=None, text="", content=b""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text
        self.content = content

    def json(self):
        return self._json


class FakeHTTP:
    def __init__(self, posts, gets):
        self.posts = list(posts)
        self.gets = list(gets)
        self.post_calls = 0
        self.get_calls = 0

    def post(self, url, **kw):
        self.post_calls += 1
        return self.posts.pop(0)

    def get(self, url, **kw):
        self.get_calls += 1
        return self.gets.pop(0)


def _engine(monkeypatch, http):
    monkeypatch.setattr(engine_mod, "requests", http)
    e = GmicloudEngine(api_key="k", poll_interval_secs=0.001, poll_attempts=3)
    e.queue = _queue.Queue()
    return e


def test_sync_outcome_normalized(monkeypatch):
    wav32k_stereo = _wav(struct.pack("<4h", 1000, 3000, 2000, 4000), 32000, 2)
    http = FakeHTTP(
        posts=[FakeResp(200, {"status": "success", "request_id": "r",
                              "outcome": {"audio_url": "https://a/x.wav"}})],
        gets=[FakeResp(200, content=wav32k_stereo)],
    )
    e = _engine(monkeypatch, http)
    assert e.synthesize("hello") is True
    pcm = e.queue.get_nowait()
    # stereo 2 frames -> mono 2 samples at 32k -> resampled to 48k (3 samples)
    assert len(pcm) >= 4
    assert len(pcm) % 2 == 0


def test_queued_polling_resolves(monkeypatch):
    wav = _wav(b"\x01\x00" * 480, 48000, 1)
    http = FakeHTTP(
        posts=[FakeResp(200, {"status": "dispatched", "request_id": "r"})],
        gets=[
            FakeResp(200, {"status": "processing", "request_id": "r"}),
            FakeResp(200, {"status": "success", "request_id": "r",
                           "outcome": {"audio_url": "https://a/x.wav"}}),
            FakeResp(200, content=wav),
        ],
    )
    e = _engine(monkeypatch, http)
    assert e.synthesize("hello") is True
    assert e.queue.get_nowait() == b"\x01\x00" * 480


def test_poll_permanent_error_fails_fast(monkeypatch):
    http = FakeHTTP(
        posts=[FakeResp(200, {"status": "dispatched", "request_id": "r"})],
        gets=[FakeResp(401, text="unauthorized")],
    )
    e = _engine(monkeypatch, http)
    assert e.synthesize("hello") is False
    assert http.get_calls == 1  # no polling after a permanent error


def test_poll_transient_retries_then_succeeds(monkeypatch):
    wav = _wav(b"\x01\x00" * 480, 48000, 1)
    http = FakeHTTP(
        posts=[FakeResp(200, {"status": "dispatched", "request_id": "r"})],
        gets=[
            FakeResp(503, text="busy"),
            FakeResp(429, text="rate limit"),
            FakeResp(200, {"status": "success", "request_id": "r",
                           "outcome": {"audio_url": "https://a/x.wav"}}),
            FakeResp(200, content=wav),
        ],
    )
    e = _engine(monkeypatch, http)
    assert e.synthesize("hello") is True


def test_failed_status_returns_false(monkeypatch):
    http = FakeHTTP(
        posts=[FakeResp(200, {"status": "failed", "request_id": "r"})],
        gets=[],
    )
    e = _engine(monkeypatch, http)
    assert e.synthesize("hello") is False


def test_http_error_returns_false(monkeypatch):
    http = FakeHTTP(posts=[FakeResp(500, text="boom")], gets=[])
    e = _engine(monkeypatch, http)
    assert e.synthesize("hello") is False


def test_empty_text_is_noop(monkeypatch):
    http = FakeHTTP(posts=[], gets=[])
    e = _engine(monkeypatch, http)
    assert e.synthesize("   ") is True
    assert e.queue.empty()


def test_non_wav_audio_returns_false(monkeypatch):
    http = FakeHTTP(
        posts=[FakeResp(200, {"status": "success", "request_id": "r",
                              "outcome": {"audio_url": "https://a/x.wav"}})],
        gets=[FakeResp(200, content=b"junk not wav")],
    )
    e = _engine(monkeypatch, http)
    assert e.synthesize("hello") is False
