"""Installed demo and server reference integration, without model inference."""
from pathlib import Path
import sys
from types import SimpleNamespace

from RealtimeTTS import demo
from RealtimeTTS.qwen_server import VoiceRegistry
from RealtimeTTS.voices import register_bundled_voices, bundled_voice_data


def test_registry_is_populated_once_and_keeps_user_voice(tmp_path):
    registry = VoiceRegistry(tmp_path)
    register_bundled_voices(registry, "qwen")
    records = registry.records()
    assert len(records) == 10
    original = registry.get("qwen-es")
    assert registry._resolve_file(original.files[0]).read_bytes() == bundled_voice_data("qwen", "es")["ref_audio"].read_bytes()
    register_bundled_voices(registry, "qwen")
    assert registry.records() == records
    custom = registry.candidate("qwen-es", "User reference", "wav", [b"custom"])
    registry.commit(custom)
    register_bundled_voices(registry, "qwen")
    assert registry.get("qwen-es") == custom


def test_demo_uses_server_defaults_and_packaged_voices():
    from RealtimeTTS.breeze_server import build_argument_parser as breeze
    from RealtimeTTS.qwen_server import build_argument_parser as qwen
    args = demo.build_argument_parser().parse_args([])
    assert args.breeze_url.endswith(":" + str(breeze().parse_args([]).port))
    assert args.qwen_url.endswith(":" + str(qwen().parse_args([]).port))
    assert args.english_voice == "breeze-en" and args.spanish_voice == "qwen-es"
    assert qwen().parse_args([]).bundled_voices
    assert breeze().parse_args([]).bundled_voices


def test_model_downloads_are_pinned_and_cached(monkeypatch):
    from RealtimeTTS.model_assets import breeze_model, alignment_model
    calls = []
    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return "/model-cache/result"
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        hf_hub_download=download, snapshot_download=download))
    assert breeze_model() == alignment_model() == Path("/model-cache/result")
    assert calls[0][0][1] == "breeze-tts-2-q8_0.gguf"
    assert len(calls[0][1]["revision"]) == len(calls[1][1]["revision"]) == 40
    assert calls[1][1]["allow_patterns"] == ["model.int8.onnx", "tokens.txt"]
