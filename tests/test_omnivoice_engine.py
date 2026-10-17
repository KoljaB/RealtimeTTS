"""OmniVoice output normalization without loading a model or importing torch."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


class TensorOutput:
    def __init__(self, waveform):
        self.waveform = waveform
        self.cpu_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        return self

    def numpy(self):
        assert self.cpu_calls == 1
        return self.waveform


@pytest.fixture
def engine_module(monkeypatch):
    torch = ModuleType("torch")
    torch.float16 = object()
    torch.no_grad = nullcontext
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    provider = ModuleType("omnivoice")
    provider.OmniVoice = Mock()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "omnivoice", provider)
    path = Path(__file__).parents[1] / "RealtimeTTS/engines/omnivoice_engine.py"
    spec = importlib.util.spec_from_file_location(
        "RealtimeTTS.engines._omnivoice_contract", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tensor", [False, True], ids=["numpy", "tensor"])
@pytest.mark.parametrize("channel_dimension", [False, True], ids=["mono", "channel-axis"])
def test_provider_output_is_queued_as_clipped_mono_pcm(
    engine_module, tensor, channel_dimension
):
    waveform = np.array([-1.2, -0.5, 0.0, 0.5, 1.2], dtype=np.float32)
    if channel_dimension:
        waveform = waveform[np.newaxis, :]
    output = TensorOutput(waveform) if tensor else waveform
    model = Mock()
    model.generate.return_value = [output]
    engine_module.OmniVoice.from_pretrained.return_value = model
    voice = engine_module.OmniVoiceVoice("reference", "reference.wav", "Reference text.")
    engine = engine_module.OmniVoiceEngine(voice=voice, device_map="cpu")
    model.generate.reset_mock()

    assert engine.synthesize("Hello from OmniVoice.", sentence_count=2)
    expected = np.array([-32767, -16383, 0, 16383, 32767], dtype="<i2")
    assert engine.queue.get_nowait() == expected.tobytes()
    assert engine.queue.empty()
    assert engine.get_stream_info() == (engine_module.pyaudio.paInt16, 1, 24000)
    model.generate.assert_called_once_with(
        language="en",
        text="Hello from OmniVoice.",
        ref_audio="reference.wav",
        ref_text="Reference text.",
        num_step=32,
        preprocess_prompt=True,
        postprocess_output=True,
    )
    if tensor:
        assert output.cpu_calls == 1
