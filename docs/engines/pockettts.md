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

## Source Notes

- Built-in source voices are `alba`, `marius`, `javert`, `jean`, `fantine`,
  `cosette`, `eponine`, and `azelma`.
- `PocketTTSVoice(name, audio_prompt_path=None)` treats a built-in name without
  a prompt as a built-in voice.
- The model is loaded with `TTSModel.load_model()`.
- Voice states are cached by voice name and prompt path.
- Output is mono 16-bit PCM at the model sample rate, falling back to 24000 Hz.

## Troubleshooting

- `pocket-tts is not installed`: install `pocket-tts` in the active environment.
- Unknown voice errors mean the name is not built in and no prompt WAV was
  supplied.
