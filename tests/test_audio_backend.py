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


def test_inflect_extras_use_standard_pyaudio_requirements():
    setup_text = _setup_text()
    assert '"inflect": standard_requirements + inflect_requirements' in setup_text
    assert (
        '"inflect-pytorch": standard_requirements + inflect_pytorch_requirements'
        in setup_text
    )
    assert '"inflect-onnx": standard_requirements + inflect_onnx_requirements' in setup_text
