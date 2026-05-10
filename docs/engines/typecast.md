# Typecast Engine

`TypecastEngine` uses the Typecast API through the `typecast-python` SDK. It is
a cloud engine and needs a Typecast API key and voice ID.

## Install

Install RealtimeTTS with the Typecast extra:

```bash
pip install "realtimetts[typecast]"
```

Set credentials in the environment or pass them to the constructor:

```powershell
$env:TYPECAST_API_KEY = "..."
$env:TYPECAST_VOICE_ID = "..."
```

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, TypecastEngine


if __name__ == "__main__":
    engine = TypecastEngine(voice_id="your-typecast-voice-id")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Typecast.")
    stream.play()
```

## Source Notes

- The engine imports `typecast-python` lazily when the client is created.
- `TYPECAST_API_KEY` is required unless `api_key` is passed.
- `TYPECAST_VOICE_ID` is used when `voice_id` is not passed.
- Default model is `ssfm-v30`; `ssfm-v21` is also accepted by the source.
- `tempo`, `pitch`, `volume`, `language`, `seed`, and emotion settings are
  forwarded into the Typecast request when provided.
- Output is WAV decoded to mono 16-bit PCM at 44100 Hz.
- `get_voices()` uses the SDK voice list and returns `TypecastVoice` objects.

## Troubleshooting

- If authentication fails, verify `TYPECAST_API_KEY` in the same shell that runs
  Python.
- If synthesis returns false, ensure a Typecast voice ID is set with either the
  constructor or `TYPECAST_VOICE_ID`.
