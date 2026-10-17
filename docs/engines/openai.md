# OpenAI Engine

`OpenAIEngine` calls the OpenAI speech API or an OpenAI-compatible speech
endpoint. Supply credentials for the chosen provider.

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

## Compatible endpoints

Pass `api_url` to select the endpoint for one engine. Include the API version
prefix (usually `/v1`); the SDK appends `/audio/speech`.

```python
engine = OpenAIEngine(
    api_url="http://localhost:8000/v1",
    api_key="your-provider-key",
    model="your-provider-model",
    voice="your-provider-voice",
    response_format="pcm",
)
```

When `api_url` is omitted, the SDK uses `OPENAI_BASE_URL` if set, otherwise
the standard OpenAI endpoint. An explicit URL overrides that environment
variable for this engine only. The provider must implement the speech endpoint
and the requested model, voice, parameters, and audio format.

## Streaming audio

Both formats stream audio as HTTP data arrives, without waiting for the whole
response. PCM is mono signed 16-bit little-endian audio at 24000 Hz. The engine
keeps incomplete samples between network chunks.

MP3 remains the default. `TextToAudioStream` uses a continuous PyAV decoder,
included in the `openai` extra, before playback, audio callbacks, and WAV export.
It outputs mono PCM at 24000 Hz and resets the decoder for each speech request.
This path does not need an external `mpv` or FFmpeg executable.

The response is closed on completion, read failure, or cancellation. A stop is
observed between received chunks; `timeout` limits blocked network reads.

## Source Notes

- Constructor defaults are `model="tts-1"`, `voice="nova"`,
  `response_format="mp3"`, and `api_key=os.getenv("OPENAI_API_KEY")`.
- Source voice names are `alloy`, `ash`, `coral`, `echo`, `fable`, `onyx`,
  `nova`, `sage`, and `shimmer`.
- `instructions`, `speed`, and `timeout` are forwarded when provided.
- `response_format` must be `mp3` or `pcm`.
- Both formats use 24000 Hz playback. Models, voices, and endpoint support
  depend on the selected provider.

## Troubleshooting

- Use `response_format="pcm"` if you want the normal PyAudio PCM path.
- For a missing MP3 decoder dependency, reinstall `realtimetts[openai]` or
  install `av>=14,<20` in the same environment.
