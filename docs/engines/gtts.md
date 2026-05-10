# gTTS Engine

`GTTSEngine` uses the Google Translate Text-to-Speech package. It is a simple
network-backed engine with no API key, but it is not an offline or commercial
service contract.

## Install

```bash
pip install "realtimetts[gtts]"
```

The source uses `gtts`, `pydub`, and a temporary WAV file before queueing PCM.
Make sure your environment can decode MP3 through the audio stack used by
`pydub`.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, GTTSEngine, GTTSVoice


if __name__ == "__main__":
    voice = GTTSVoice(language="en", tld="com", speed=1.0)
    stream = TextToAudioStream(GTTSEngine(voice))
    stream.feed("Hello from gTTS.")
    stream.play()
```

## Source Notes

- `GTTSVoice(language="en", tld="com", chunk_length=100,
  crossfade_length=10, speed=1.0)` controls language, regional TLD, and optional
  speedup.
- Passing a string to `set_voice()` creates `GTTSVoice(language=that_string,
  tld="com")`.
- `get_voices()` builds combinations from `gtts.lang.tts_langs()` and common
  TLDs.
- Output is queued as mono 16-bit PCM at 22050 Hz after MP3-to-WAV conversion.

## Troubleshooting

- Network failures or service changes surface as synthesis errors.
- If speedup sounds choppy, keep `speed=1.0` or tune `chunk_length` and
  `crossfade_length`.
