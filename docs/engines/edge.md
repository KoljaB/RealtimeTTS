# Edge Engine

`EdgeEngine` uses the `edge-tts` package and Microsoft Edge online voices. It is
a free service wrapper, not an offline engine.

## Install

```bash
pip install "realtimetts[edge]"
```

The engine returns compressed audio chunks and reports a custom PyAudio format.
For local playback, install `mpv` as described in
[output and files](../output-and-files.md).

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, EdgeEngine


if __name__ == "__main__":
    engine = EdgeEngine(rate=0, pitch=0, volume=0)
    engine.set_voice("en-US-EmmaMultilingualNeural")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Edge TTS.")
    stream.play()
```

## Source Notes

- `EdgeVoice` stores `ShortName`, full name, gender, locale, status, codec, and
  voice tags returned by `edge_tts.list_voices()`.
- `EdgeEngine(rate=0, pitch=0, volume=0)` formats those controls as Edge TTS
  percentage or Hertz values.
- If no voice is set before synthesis, the source selects
  `en-US-EmmaMultilingualNeural`.
- `set_voice()` first tries exact voice-name match, then substring and
  case-insensitive matches.

## Troubleshooting

- `get_voices()` calls the online Edge voice listing API and can take time.
- Compressed audio playback depends on `mpv` for normal local speaker output.
