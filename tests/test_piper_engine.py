import subprocess

import pytest

from RealtimeTTS import PiperEngine, PiperVoice


@pytest.mark.parametrize("speaker_id", [None, 0, 7])
def test_synthesize_selects_the_requested_piper_speaker(monkeypatch, speaker_id):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout=b"\x00\x00")

    monkeypatch.setattr(subprocess, "run", run)
    voice = PiperVoice("voice.onnx", "voice.onnx.json", speaker_id=speaker_id)
    engine = PiperEngine(piper_path="piper", voice=voice)

    assert engine.synthesize("Hello")
    args, kwargs = calls[0]
    assert args[args.index("-m") + 1] == "voice.onnx"
    assert args[args.index("-c") + 1] == "voice.onnx.json"
    if speaker_id is None:
        assert "--speaker" not in args
    else:
        assert args[args.index("--speaker") + 1] == str(speaker_id)
    assert kwargs["input"] == b"Hello"
    assert kwargs["shell"] is False
    assert engine.queue.get_nowait() == b"\x00\x00"


def test_set_voice_changes_the_selected_speaker(monkeypatch):
    speakers = []

    def run(args, **kwargs):
        speakers.append(args[args.index("--speaker") + 1])
        return subprocess.CompletedProcess(args, 0, stdout=b"\x00\x00")

    monkeypatch.setattr(subprocess, "run", run)
    engine = PiperEngine(voice=PiperVoice("voice.onnx", speaker_id=0))
    assert engine.synthesize("First speaker")
    engine.set_voice(PiperVoice("voice.onnx", speaker_id=3))
    assert engine.synthesize("Second speaker")
    assert speakers == ["0", "3"]
