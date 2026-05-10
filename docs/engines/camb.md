# CAMB Engine

`CambEngine` uses the CAMB AI MARS text-to-speech API through `camb-sdk`.

## Install

```bash
pip install "realtimetts[camb]"
```

Set the API key in the environment or pass it to the constructor:

```powershell
$env:CAMB_API_KEY = "..."
```

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, CambEngine


if __name__ == "__main__":
    engine = CambEngine(voice_id=147320, language="en-us")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from CAMB.")
    stream.play()
```

## Source Notes

- Constructor defaults are `voice_id=147320`, `language="en-us"`,
  `speech_model="mars-flash"`, and `output_format="wav"`.
- `speech_model` can be `mars-flash`, `mars-pro`, or `mars-instruct` according
  to the source docstring.
- `user_instructions` is only sent when using `mars-instruct`.
- `get_voices()` currently returns an empty list. Use numeric voice IDs or
  `CambVoice`.
- The engine reports a custom stream format because API output can be WAV or MP3.

## Troubleshooting

- Missing key errors can be fixed by passing `api_key` or setting
  `CAMB_API_KEY`.
- If instructions are ignored, check that `speech_model="mars-instruct"` is set.
