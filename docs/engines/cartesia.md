# Cartesia Engine

`CartesiaEngine` uses the Cartesia API over its WebSocket TTS connection.

## Install

Install RealtimeTTS with the Cartesia extra:

```bash
pip install "realtimetts[cartesia]"
```

Set the API key in the environment or pass it to the constructor:

```powershell
$env:CARTESIA_API_KEY = "..."
```

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, CartesiaEngine


if __name__ == "__main__":
    engine = CartesiaEngine(voice_id="your-cartesia-voice-id")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Cartesia.")
    stream.play()
```

## Source Notes

- Default `model_id` is `sonic-3`; valid model IDs in source include
  `sonic-2`, `sonic-3`, `sonic-turbo`, and `sonic`.
- Default output is raw `pcm_f32le` at 44100 Hz. `pcm_s16le` is also supported.
- `language` and `add_timestamps` are optional request fields.
- `CartesiaVoice(id, name=None, language=None)` can be used for voice selection.
- `get_voices()` fetches one page by default. Set `fetch_all_voices=True` for
  full pagination.

## Troubleshooting

- `Missing Cartesia API key`: pass `api_key` or set `CARTESIA_API_KEY`.
- `No voice set`: pass `voice_id`, a `CartesiaVoice`, or a dict with `id`.
