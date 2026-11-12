"""AzureEngine output-format contract tests.

The Azure Speech SDK and PyAudio are replaced with local fakes, so these tests
need neither the optional SDK, an audio device, nor network access.
"""

import enum
import importlib
import queue
import sys
import types
from datetime import timedelta

import pytest

_FORMAT_MEMBERS = [
    "Riff8Khz16BitMonoPcm",
    "Riff16Khz16BitMonoPcm",
    "Riff22050Hz16BitMonoPcm",
    "Riff24Khz16BitMonoPcm",
    "Riff44100Hz16BitMonoPcm",
    "Riff48Khz16BitMonoPcm",
    "Raw8Khz16BitMonoPcm",
    "Raw16Khz16BitMonoPcm",
    "Raw22050Hz16BitMonoPcm",
    "Raw24Khz16BitMonoPcm",
    "Raw44100Hz16BitMonoPcm",
    "Raw48Khz16BitMonoPcm",
]

EXPECTED_FORMATS = {
    "riff-8khz-16bit-mono-pcm": ("Riff8Khz16BitMonoPcm", 8000),
    "riff-16khz-16bit-mono-pcm": ("Riff16Khz16BitMonoPcm", 16000),
    "riff-22050hz-16bit-mono-pcm": ("Riff22050Hz16BitMonoPcm", 22050),
    "riff-24khz-16bit-mono-pcm": ("Riff24Khz16BitMonoPcm", 24000),
    "riff-44100hz-16bit-mono-pcm": ("Riff44100Hz16BitMonoPcm", 44100),
    "riff-48khz-16bit-mono-pcm": ("Riff48Khz16BitMonoPcm", 48000),
    "raw-8khz-16bit-mono-pcm": ("Raw8Khz16BitMonoPcm", 8000),
    "raw-16khz-16bit-mono-pcm": ("Raw16Khz16BitMonoPcm", 16000),
    "raw-22050hz-16bit-mono-pcm": ("Raw22050Hz16BitMonoPcm", 22050),
    "raw-24khz-16bit-mono-pcm": ("Raw24Khz16BitMonoPcm", 24000),
    "raw-44100hz-16bit-mono-pcm": ("Raw44100Hz16BitMonoPcm", 44100),
    "raw-48khz-16bit-mono-pcm": ("Raw48Khz16BitMonoPcm", 48000),
}

PA_INT16 = 8


def _fake_speech_sdk():
    """Build a minimal stand-in for azure.cognitiveservices.speech."""
    speech = types.ModuleType("azure.cognitiveservices.speech")
    speech.SpeechSynthesisOutputFormat = enum.Enum(
        "SpeechSynthesisOutputFormat", _FORMAT_MEMBERS
    )
    speech.ResultReason = enum.Enum(
        "ResultReason", ["SynthesizingAudioCompleted", "Canceled"]
    )
    speech.CancellationReason = enum.Enum("CancellationReason", ["Error"])
    speech.synthesizers = []

    class SpeechConfig:
        def __init__(self, subscription, region):
            self.subscription = subscription
            self.region = region
            self.output_format = None

        def set_speech_synthesis_output_format(self, output_format):
            self.output_format = output_format

    class _Signal:
        def __init__(self):
            self.handlers = []

        def connect(self, handler):
            self.handlers.append(handler)

    class SpeechSynthesizer:
        def __init__(self, speech_config, audio_config):
            self.speech_config = speech_config
            self.audio_config = audio_config
            self.synthesis_word_boundary = _Signal()
            self.ssml = None
            speech.synthesizers.append(self)

        def speak_ssml_async(self, ssml):
            self.ssml = ssml
            # Deliver audio through the push stream like the real SDK does.
            self.audio_config.stream.callback.write(memoryview(b"\x01\x00\x02\x00"))
            result = types.SimpleNamespace(
                reason=speech.ResultReason.SynthesizingAudioCompleted,
                audio_duration=timedelta(milliseconds=250),
            )
            return types.SimpleNamespace(get=lambda: result)

    audio = types.ModuleType("azure.cognitiveservices.speech.audio")

    class PushAudioOutputStreamCallback:
        pass

    class PushAudioOutputStream:
        def __init__(self, callback):
            self.callback = callback

    class AudioOutputConfig:
        def __init__(self, stream):
            self.stream = stream

    audio.PushAudioOutputStreamCallback = PushAudioOutputStreamCallback
    audio.PushAudioOutputStream = PushAudioOutputStream
    audio.AudioOutputConfig = AudioOutputConfig

    speech.SpeechConfig = SpeechConfig
    speech.SpeechSynthesizer = SpeechSynthesizer
    speech.audio = audio
    return speech, audio


@pytest.fixture
def azure(monkeypatch):
    speech, audio = _fake_speech_sdk()
    azure_pkg = types.ModuleType("azure")
    cognitive = types.ModuleType("azure.cognitiveservices")
    azure_pkg.cognitiveservices = cognitive
    cognitive.speech = speech
    pyaudio_stub = types.ModuleType("pyaudio")
    pyaudio_stub.paInt16 = PA_INT16

    monkeypatch.setitem(sys.modules, "azure", azure_pkg)
    monkeypatch.setitem(sys.modules, "azure.cognitiveservices", cognitive)
    monkeypatch.setitem(sys.modules, "azure.cognitiveservices.speech", speech)
    monkeypatch.setitem(sys.modules, "azure.cognitiveservices.speech.audio", audio)
    monkeypatch.setitem(sys.modules, "pyaudio", pyaudio_stub)
    monkeypatch.delitem(sys.modules, "RealtimeTTS.engines.azure_engine", raising=False)

    module = importlib.import_module("RealtimeTTS.engines.azure_engine")
    yield module, speech
    # Do not leak a module bound to the fake SDK into other tests.
    sys.modules.pop("RealtimeTTS.engines.azure_engine", None)
    engines = sys.modules.get("RealtimeTTS.engines")
    if getattr(engines, "azure_engine", None) is module:
        delattr(engines, "azure_engine")


def test_supported_formats_cover_all_16bit_mono_pcm_rates(azure):
    module, speech = azure
    engine_cls = module.AzureEngine

    assert set(engine_cls.SUPPORTED_AUDIO_FORMATS) == set(EXPECTED_FORMATS)
    assert set(engine_cls.AUDIO_FORMAT_MAP) == set(EXPECTED_FORMATS)
    for name, (member, rate) in EXPECTED_FORMATS.items():
        assert engine_cls.SUPPORTED_AUDIO_FORMATS[name] == rate
        assert (
            engine_cls.AUDIO_FORMAT_MAP[name]
            is speech.SpeechSynthesisOutputFormat[member]
        )


def test_default_format_is_unchanged(azure):
    module, speech = azure
    engine = module.AzureEngine("key", "eastus")

    assert engine.audio_format == "riff-16khz-16bit-mono-pcm"
    assert engine.get_stream_info() == (PA_INT16, 1, 16000)


@pytest.mark.parametrize("audio_format", sorted(EXPECTED_FORMATS))
def test_stream_info_and_sdk_format_follow_audio_format(azure, audio_format):
    module, speech = azure
    member, rate = EXPECTED_FORMATS[audio_format]
    engine = module.AzureEngine("key", "eastus", audio_format=audio_format)
    engine.queue = queue.Queue()

    assert engine.get_stream_info() == (PA_INT16, 1, rate)
    assert engine.synthesize("Hello there.") is True

    synthesizer = speech.synthesizers[-1]
    assert synthesizer.speech_config.subscription == "key"
    assert synthesizer.speech_config.region == "eastus"
    assert (
        synthesizer.speech_config.output_format
        is speech.SpeechSynthesisOutputFormat[member]
    )
    assert synthesizer.audio_config.stream.callback.sample_rate == rate
    assert engine.queue.get_nowait() == b"\x01\x00\x02\x00"


@pytest.mark.parametrize(
    "audio_format",
    ["riff-8khz-8bit-mono-mulaw", "RAW-8KHZ-16BIT-MONO-PCM", "", "pcm"],
)
def test_unsupported_audio_format_is_rejected(azure, audio_format):
    module, _ = azure

    with pytest.raises(ValueError, match="Invalid audio_format"):
        module.AzureEngine("key", "eastus", audio_format=audio_format)
