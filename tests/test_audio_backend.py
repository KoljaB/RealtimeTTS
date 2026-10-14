from pathlib import Path

from RealtimeTTS import _audio_backend as backend


def _setup_text() -> str:
    return (Path(__file__).parents[1] / "setup.py").read_text(encoding="utf-8")


def test_supported_audio_backend_is_pyaudio():
    assert backend.AUDIO_BACKEND == "pyaudio"
    assert backend.pyaudio.PyAudio
    assert backend.pa


def test_qwen_extra_uses_pyaudio_and_not_miniaudio():
    setup_text = _setup_text()
    assert (
        "qwen_requirements = qwen_common_requirements + pyaudio_requirements"
        in setup_text
    )
    assert '"qwen": base_requirements + qwen_requirements' in setup_text
    assert "miniaudio" not in setup_text


def test_pcm_format_values_match_the_playback_abi():
    from RealtimeTTS import _audio_formats as formats
    for name, native in (
        ("FLOAT32", "paFloat32"), ("INT32", "paInt32"), ("INT24", "paInt24"),
        ("INT16", "paInt16"), ("INT8", "paInt8"), ("UINT8", "paUInt8"),
        ("CUSTOM", "paCustomFormat"),
    ):
        assert getattr(formats, name) == getattr(backend.pyaudio, native)


def test_headless_servers_and_alignment_do_not_load_local_playback():
    import subprocess
    import sys
    script = """
import importlib.abc
import sys
class NoPlayback(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyaudio' or fullname.startswith('pyaudio.') or fullname == 'RealtimeTTS._audio_backend':
            raise AssertionError('Headless path imported local playback: ' + fullname)
sys.meta_path.insert(0, NoPlayback())
from types import SimpleNamespace
from RealtimeTTS import _audio_formats as formats
from RealtimeTTS.alignment.audio import audio_bytes_to_mono_float32
from RealtimeTTS.alignment.server import parse_alignment_request
from RealtimeTTS.qwen_server import QwenHttpServer
from RealtimeTTS.server import GenericServer
from RealtimeTTS.engines.qwen_engine import QwenEngine
assert QwenEngine.get_stream_info(None) == (formats.INT16, 1, 24000)
samples, rate = audio_bytes_to_mono_float32(bytes([0, 64, 0, 192]), formats.INT16, 1, 24000)
assert list(samples) == [.5, -.5] and rate == 24000
engine = SimpleNamespace(provides_word_timings=True, streaming_aligner=None)
assert parse_alignment_request(None, engine, 'english') is None
assert parse_alignment_request({'source': 'native'}, engine, 'english').source == 'native'
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_inflect_extras_use_standard_pyaudio_requirements():
    setup_text = _setup_text()
    assert '"inflect": standard_requirements + inflect_requirements' in setup_text
    assert (
        '"inflect-pytorch": standard_requirements + inflect_pytorch_requirements'
        in setup_text
    )
    assert '"inflect-onnx": standard_requirements + inflect_onnx_requirements' in setup_text
