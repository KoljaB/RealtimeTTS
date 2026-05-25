# PocketTTS Engine

`PocketTTSEngine` wraps Kyutai Labs' Pocket TTS package. The source describes it
as a lightweight CPU-oriented English model with optional voice cloning.

## Install

Install RealtimeTTS with the PocketTTS extra:

```bash
pip install "realtimetts[pockettts]"
```

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, PocketTTSEngine


if __name__ == "__main__":
    engine = PocketTTSEngine(voice="alba")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Pocket TTS.")
    stream.play()
```

Voice cloning:

```python
from RealtimeTTS import PocketTTSVoice, PocketTTSEngine

voice = PocketTTSVoice(name="custom", audio_prompt_path="reference.wav")
engine = PocketTTSEngine(voice=voice)
```

## CPU and GPU

PocketTTS is CPU-oriented, and `PocketTTSEngine` runs on CPU by default:

```python
engine = PocketTTSEngine(voice="alba")
```

This is equivalent to:

```python
engine = PocketTTSEngine(voice="alba", device="cpu")
```

To try CUDA, install a CUDA-enabled PyTorch build first, then pass a CUDA device:

```python
engine = PocketTTSEngine(voice="alba", device="cuda")
```

You can also use a concrete device such as `device="cuda:0"`. If CUDA is not
available in the active PyTorch install, initialization fails from PyTorch; use
`device="cpu"` in that environment. Voice states are created after the model is
moved to the selected device, so built-in voices and cloned voices use the same
device.

## Source Notes

- Built-in source voices are `alba`, `marius`, `javert`, `jean`, `fantine`,
  `cosette`, `eponine`, and `azelma`.
- `PocketTTSVoice(name, audio_prompt_path=None)` treats a built-in name without
  a prompt as a built-in voice.
- The model is loaded with `TTSModel.load_model()` and then moved to the
  selected Torch device.
- Voice states are cached by voice name and prompt path.
- Output is mono 16-bit PCM at the model sample rate, falling back to 24000 Hz.

## Runtime Notes

`PocketTTSEngine` keeps Pocket model work on a persistent synthesis worker and
patches Pocket's short-text streaming path to decode serially. This avoids a
Windows/Torch CPU memory-retention issue observed when repeated short
generations create fresh synthesis and decoder threads. The public
`TextToAudioStream` API remains unchanged; the worker is an internal engine
detail and is stopped by `shutdown()`.

## Troubleshooting

- `pocket-tts is not installed`: install `pocket-tts` in the active environment.
- Unknown voice errors mean the name is not built in and no prompt WAV was
  supplied.
