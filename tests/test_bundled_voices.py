"""Installed reference data and engine selection must work without a model."""
import hashlib
import json
import wave
from importlib import resources
from pathlib import Path

import pytest

from RealtimeTTS import BreezeEngine, BreezeVoice, QwenEngine, QwenVoice
from RealtimeTTS.voices import bundled_voice_data, bundled_voices_data


def test_catalog_contains_exact_audio_and_transcripts():
    catalog = json.loads(resources.files("RealtimeTTS.voices").joinpath("catalog.json").read_text(encoding="utf-8"))
    assert len(catalog["voices"]) == 12
    assert len({entry["name"] for entry in catalog["voices"]}) == 12
    for entry in catalog["voices"]:
        voice = bundled_voice_data(entry["engine"], entry["language_code"])
        wav_path = Path(voice["ref_audio"])
        assert hashlib.sha256(wav_path.read_bytes()).hexdigest() == entry["sha256"]
        assert voice["ref_text"].strip() == entry["ref_text"]
        assert voice["language"] == entry["language"]
        with wave.open(str(wav_path), "rb") as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000)
            assert wav.getnframes() / wav.getframerate() == pytest.approx(entry["duration_s"], abs=0.001)


@pytest.mark.parametrize("code", ("en", "es", "de", "fr", "pt", "ru", "it", "ja", "zh", "ko"))
def test_qwen_factory_and_base_engine_discovery(code):
    voice = QwenVoice.builtin(code)
    assert voice.name == f"qwen-{code}"
    assert voice.clone_mode == "icl"
    assert Path(voice.ref_audio).is_file()
    engine = object.__new__(QwenEngine)
    engine.model_type = "base"
    assert voice in engine.get_voices()
    engine.model_type = "voice_design"
    assert engine.get_voices() == []


def test_factory_aliases_errors_and_independent_voice_objects():
    assert QwenVoice.builtin("german") == QwenVoice.builtin("qwen-de")
    assert BreezeVoice.builtin("chinese") == BreezeVoice.builtin("breeze-zh")
    voice = QwenVoice.builtin("en")
    voice.ref_text = "changed locally"
    assert QwenVoice.builtin("en").ref_text != voice.ref_text
    with pytest.raises(ValueError, match="available languages: en, zh"):
        BreezeVoice.builtin("de")
    with pytest.raises(ValueError, match="No bundled qwen voice"):
        QwenVoice.builtin("../missing")
    assert len(bundled_voices_data("qwen")) == 10


@pytest.mark.parametrize("code", ("en", "zh"))
def test_breeze_lists_and_uploads_packaged_voice(monkeypatch, code):
    import RealtimeTTS.engines.breeze_engine as module

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
            return Response([{"id": "existing", "ref_text": "Existing voice."}])
        def post(self, url, **kwargs):
            self.posts.append(kwargs)
            return Response({"id": kwargs["data"]["name"]})
        def close(self):
            pass

    http = Http()
    monkeypatch.setattr(module.requests, "Session", lambda: http)
    engine = BreezeEngine()
    try:
        voices = engine.get_voices()
        assert voices[0].voice_id == "existing"
        voice = next(voice for voice in voices if voice.name == f"breeze-{code}")
        assert voice == BreezeVoice.builtin(code)
        engine.set_voice(voice)
        assert len(http.posts) == 1
        assert http.posts[0]["files"]["ref_audio"][1] == Path(voice.ref_audio).read_bytes()
        assert http.posts[0]["data"]["ref_text"] == voice.ref_text
        assert engine.get_language() == voice.language
    finally:
        engine.shutdown()
