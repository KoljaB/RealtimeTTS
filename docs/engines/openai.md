# OpenAI Engine

`OpenAIEngine` calls the OpenAI speech API. It is a cloud engine and requires an
OpenAI API key.

## Install

```bash
pip install "realtimetts[openai]"
```

Set the API key in the environment or pass it to the constructor:

```powershell
$env:OPENAI_API_KEY = "..."
```

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, OpenAIEngine


if __name__ == "__main__":
    engine = OpenAIEngine(model="tts-1", voice="nova", response_format="pcm")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from OpenAI text to speech.")
    stream.play()
```

## Source Notes

- Constructor defaults are `model="tts-1"`, `voice="nova"`,
  `response_format="mp3"`, and `api_key=os.getenv("OPENAI_API_KEY")`.
- Source voice names are `alloy`, `ash`, `coral`, `echo`, `fable`, `onyx`,
  `nova`, `sage`, and `shimmer`.
- `instructions`, `speed`, and `timeout` are forwarded when provided.
- `response_format` must be `mp3` or `pcm`.
- PCM output reports mono 16-bit PCM at 22050 Hz. MP3 output uses custom-format
  playback and normally needs `mpv`.

## Troubleshooting

- Use `response_format="pcm"` if you want the normal PyAudio PCM path.
- MP3 playback issues usually mean `mpv` is missing or not visible on `PATH`.
