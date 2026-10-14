from .base import (
    CharacterAligner,
    CharacterAlignment,
    SpeechSegment,
    VoiceActivityDetector,
    WordAlignment,
    WordAligner,
)
from .timeline import AlignmentConfig, AlignedSpan, AlignmentSnapshot, AlignmentTimeline
from .streaming import StreamingTextAligner
from .evidence import CompletionPolicy
from .onnx_ctc import OnnxCTCBackend
from .fake import FakeWordAligner
# Optional offline/playback integrations must not load PyAudio on a headless
# server that only requests the CPU streaming aligner.
def __getattr__(name):
    from importlib import import_module

    modules = {
        "OmniASRCTCAligner": "omniasr",
        "TorchaudioMMSFACharacterAligner": "torchaudio_fa",
        "SileroVoiceActivityDetector": "vad",
        "WavTranscriptCharacterAligner": "wav_transcript",
        "align_wav_transcript_characters": "wav_transcript",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module("." + modules[name], __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "AlignmentConfig", "AlignedSpan", "AlignmentSnapshot", "AlignmentTimeline",
    "StreamingTextAligner", "CompletionPolicy",
    "OnnxCTCBackend",
    "CharacterAligner",
    "CharacterAlignment",
    "FakeWordAligner",
    "OmniASRCTCAligner",
    "SileroVoiceActivityDetector",
    "SpeechSegment",
    "TorchaudioMMSFACharacterAligner",
    "VoiceActivityDetector",
    "WavTranscriptCharacterAligner",
    "WordAlignment",
    "WordAligner",
    "align_wav_transcript_characters",
]
