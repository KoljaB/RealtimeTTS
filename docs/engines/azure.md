# Azure Engine

`AzureEngine` uses Azure Cognitive Services Speech. It is a cloud engine with
voice selection, SSML rate/pitch controls, emotions, and word timing support.

## Install

```bash
pip install "realtimetts[azure]"
```

Pass credentials to the constructor. The older docs also mention environment
variables `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION`, but the current source
constructor takes `speech_key` and `service_region` directly.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, AzureEngine


if __name__ == "__main__":
    engine = AzureEngine(
        speech_key="...",
        service_region="eastus",
        voice="en-US-AshleyNeural",
    )
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Azure.")
    stream.play()
```

## Source Notes

- Supported output formats are `riff-16khz-16bit-mono-pcm`,
  `riff-24khz-16bit-mono-pcm`, and `riff-48khz-16bit-mono-pcm`.
- `get_stream_info()` returns mono 16-bit PCM at the selected Azure sample rate.
- `get_voices()` fetches Azure voices and wraps them as
  `AzureVoice(name, locale, gender)`.
- `set_voice_parameters()` supports fields such as `rate`, `pitch`, `emotion`,
  `emotion_degree`, and `emotion_role`.
- Word boundary events are converted into RealtimeTTS `TimingInfo` entries.

## Troubleshooting

- Check that the region matches the Azure Speech resource region.
- If an emotion style does not take effect, verify that the selected Azure voice
  supports that style.
