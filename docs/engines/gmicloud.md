# GMI Cloud Engine

`GmicloudEngine` uses GMI Cloud's multimodal inference request queue with
family-aware request mapping for the speech models it hosts: Inworld Realtime
TTS (`inworld-tts-*`), MiniMax speech synthesis (`minimax-tts-speech-*`), and
Resemble AI's Chatterbox (`Chatterbox-tts`, requires a Resemble `voice_uuid`).
It returns WAV audio and requires a GMI Cloud API key.

## Install

```bash
pip install "realtimetts[gmicloud]"
```

Set the API key in the environment or pass it to the constructor:

```powershell
$env:GMI_CLOUD_API_KEY = "..."
```

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, GmicloudEngine


if __name__ == "__main__":
    engine = GmicloudEngine(model="inworld-tts-2", voice="Dennis")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from GMI Cloud.")
    stream.play()
```

Chatterbox models need a Resemble voice UUID as the voice:

```python
engine = GmicloudEngine(model="Chatterbox-tts", voice="your-resemble-voice-uuid")
```

## Source Notes

- The request shape is selected from the model id (unknown ids use the
  Inworld shape). Both documented response shapes are resolved: synchronous
  inline `outcome` responses and queued generations resolved by polling.
- Every response is normalized to mono 16-bit PCM at `sample_rate`
  (48 kHz by default), so emitted audio matches `get_stream_info()` exactly
  (MiniMax's native 32 kHz stereo output is down-mixed and resampled).
- MiniMax generations go through GMI Cloud's queue and can be slow;
  `poll_attempts` / `poll_interval_secs` are configurable. The Inworld
  family is the low-latency option.
- Permanent HTTP errors (e.g. 401/404) fail immediately while polling; only
  408/429/5xx are retried.
- `speed`, `temperature` and `extra_payload` allow request customization;
  `temperature` applies to the Inworld family.

Also published on PyPI as `realtimetts-gmicloud` for use without a local
checkout.
