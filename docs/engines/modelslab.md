# ModelsLab Engine

`ModelsLabEngine` uses the ModelsLab text-to-speech API. It exists under
`RealtimeTTS.engines`, but the root `RealtimeTTS` package does not currently
lazy-export it. Import it from `RealtimeTTS.engines` until that mismatch is
resolved.

## Install

Install RealtimeTTS with the ModelsLab extra. The source uses `requests` plus
normal RealtimeTTS playback dependencies:

```bash
pip install "realtimetts[modelslab]"
```

Set the API key in the environment or pass it to the constructor:

```powershell
$env:MODELSLAB_API_KEY = "..."
```

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream
from RealtimeTTS.engines import ModelsLabEngine


if __name__ == "__main__":
    engine = ModelsLabEngine(voice="madison", language="american english")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from ModelsLab.")
    stream.play()
```

## Source Notes

- Defaults are `voice="madison"`, `language="american english"`,
  `speed=1.0`, and `emotion=False`.
- The source includes built-in voice IDs for American/British English, Spanish,
  French, German, Italian, Japanese, Hindi, Mandarin Chinese, and Brazilian
  Portuguese.
- The API can return `processing`; the engine polls the returned fetch URL.
- Output is MP3 and reports custom-format playback at 22050 Hz.

## Troubleshooting

- Missing key errors can be fixed by passing `api_key` or setting
  `MODELSLAB_API_KEY`.
- Because the root package export is absent, prefer
  `from RealtimeTTS.engines import ModelsLabEngine`.
