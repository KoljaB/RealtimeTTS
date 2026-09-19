"""Protect the publicly advertised emotional demo and both execution paths."""
import io
import json
from pathlib import Path
import queue
import subprocess
import sys
import types
import wave

import pytest

from RealtimeTTS import qwen_emotions as demo


ROOT = Path(__file__).resolve().parents[1]


def test_advertised_script_lists_original_emotions_from_tests_directory():
    result = subprocess.run(
        [sys.executable, "-B", "faster_qwen_emotions.py", "--list"],
        cwd=ROOT / "tests", capture_output=True, text=True, check=True,
    )
    assert result.stdout.splitlines() == list(demo.ASSET_HASHES)


def test_module_help_does_not_import_native_playback_or_models():
    result = subprocess.run(
        [sys.executable, "-B", "-c",
         "import sys; from RealtimeTTS.qwen_emotions import main; "
         "main(['--list']); assert not {'qwentts_cpp','pyaudio','torch'} & sys.modules.keys()"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_original_references_are_present_and_match_pinned_downloads():
    entries = demo.build_emotion_entries(ROOT / "tests" / "ears_emotional_speaker11")
    assert len(entries) == 11
    demo.ensure_references(entries, download=True)
    assert entries[0].name == "neutral"
    assert entries[-1].name == "distress"
    assert "steady and predictable like a reliable old toaster" in entries[0].speak_text


def test_missing_explicit_references_never_silently_download(tmp_path, monkeypatch):
    monkeypatch.setattr(demo.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected network"))
    with pytest.raises(FileNotFoundError, match="Missing reference"):
        demo.ensure_references(demo.build_emotion_entries(tmp_path), download=False)


def test_download_hash_mismatch_does_not_install_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(demo.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"bad download"))
    entry = demo.build_emotion_entries(tmp_path)[0]
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        demo.ensure_references([entry], download=True)
    assert not Path(entry.ref_audio).exists()


def test_empty_synthesis_is_failure(tmp_path):
    path = tmp_path / "empty.wav"
    demo._write_pcm(path, [])
    with pytest.raises(RuntimeError, match="nonempty"):
        demo._check_wav(path)


@pytest.mark.parametrize("cpu", [False, True])
def test_native_demo_preserves_model_mode_and_voice_switching(cpu, tmp_path, monkeypatch):
    import RealtimeTTS.engines.qwen_engine as gpu_module
    import RealtimeTTS.engines.qwen_cpu_engine as cpu_module
    instances = []

    class Engine:
        def __init__(self, **options):
            instances.append(self)
            self.options = options
            self.queue = queue.Queue()
            self.voices = []
            self.last_synthesis_profile = {"first_queue_ms": 12.0}
            self.last_error = None
            self.closed = False

        def set_voice(self, voice):
            assert voice.instruct is None
            self.voices.append(voice.name)

        def warmup(self):
            pass

        def synthesize(self, text):
            assert text == "Short test."
            self.queue.put(b"\x01\x00" * 2400)
            return True

        def shutdown(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "qwentts_cpp", types.SimpleNamespace(CPU_ONLY=cpu))
    monkeypatch.setattr(gpu_module, "QwenEngine", Engine)
    monkeypatch.setattr(cpu_module, "QwenCpuEngine", Engine)
    result = demo.main([
        "--reference-dir", str(ROOT / "tests" / "ears_emotional_speaker11"),
        "--emotions", "neutral", "anger", "--no-play", "--text", "Short test.",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    engine = instances[0]
    assert engine.closed
    assert engine.options["model_id"] == demo.MODEL_ID
    assert engine.options["clone_mode"] == "speaker_only"
    assert engine.options["quant"] == "Q8_0"
    if cpu:
        assert engine.options["onset_silence_profile"] == "qwen3_tts_12hz_0_6b_base_q8_v1"
        assert engine.options["onset_silence_recovery"] is True
    else:
        assert "onset_silence_recovery" not in engine.options
    assert engine.voices[-2:] == ["neutral", "anger"]
    assert demo._check_wav(tmp_path / "neutral.wav") == .1
    assert len(json.loads((tmp_path / "results.json").read_text())) == 2


def test_server_demo_registers_wav_and_streams_pcm_without_local_engine(tmp_path, monkeypatch):
    calls = []

    def request(url, key, payload=None):
        calls.append((url, payload))
        if url.endswith("capabilities"):
            return io.BytesIO(json.dumps({"engine": {"cpu_only": True, "model_id": demo.MODEL_ID}}).encode())
        if url.endswith("voices"):
            assert "wav_b64" in payload
            assert payload["ref_text"]
            return io.BytesIO(b"{}")
        assert payload["clone_mode"] == "speaker_only"
        assert "instructions" not in payload
        assert payload["response_format"] == "pcm"
        return io.BytesIO(b"\x01\x00" * 2400)

    monkeypatch.setattr(demo, "_request", request)
    monkeypatch.setattr(demo, "run_local", lambda *a: pytest.fail("loaded local model"))
    assert demo.main([
        "--server", "http://localhost:8080", "--device", "cpu", "--no-play",
        "--reference-dir", str(ROOT / "tests" / "ears_emotional_speaker11"),
        "--emotions", "neutral", "--output-dir", str(tmp_path),
    ]) == 0
    assert len(calls) == 3
    assert demo._check_wav(tmp_path / "neutral.wav") == .1


def test_demo_is_packaged_and_console_entry_point_is_preserved():
    assert (ROOT / "RealtimeTTS" / "qwen_emotions.py").is_file()
    assert "realtimetts-qwen-emotions=RealtimeTTS.qwen_emotions:main" in (ROOT / "setup.py").read_text()
