# Speechify Engine

`SpeechifyEngine` uses the Speechify API streaming endpoint
(`POST /v1/audio/stream`). Audio arrives as raw 16-bit mono PCM while it is
generated, so playback starts before the sentence is finished.

## Install

```bash
pip install "realtimetts[speechify]"
```

Set the API key in the environment or pass it to the constructor:

```powershell
$env:SPEECHIFY_API_KEY = "..."
```

Keys are created at https://platform.speechify.ai.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, SpeechifyEngine


if __name__ == "__main__":
    engine = SpeechifyEngine(voice_id="geffen_32", model="simba-3.2")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Speechify.")
    stream.play()
```

## Source Notes

- Model defaults to `simba-3.2` (English). Use `simba-3.0` with `language`
  (for example `fr-FR`) for other languages.
- Voice defaults to `geffen_32`. `get_voices()` lists the voices available to
  your key as `SpeechifyVoice(voice_id, name, locale, gender)`.
- `sample_rate` picks the PCM format (`8000`, `16000`, `22050`, `24000`,
  `44100` or `48000`). Defaults to `24000`.
- `loudness_normalization` and `text_normalization` are passed through as API
  options. Both add some latency.
- API docs: https://docs.speechify.ai

## Troubleshooting

- `Speechify API key is required`: pass `api_key` or set `SPEECHIFY_API_KEY`.
- A 400 naming the voice usually means a non-English voice was used with
  `simba-3.2`; switch to `simba-3.0`.
