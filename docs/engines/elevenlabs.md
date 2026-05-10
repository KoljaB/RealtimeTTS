# ElevenLabs Engine

`ElevenlabsEngine` uses the ElevenLabs streaming text-to-speech API. It is a
cloud engine and requires an ElevenLabs API key.

## Install

```bash
pip install "realtimetts[elevenlabs]"
```

Set the API key in the environment or pass it to the constructor:

```powershell
$env:ELEVENLABS_API_KEY = "..."
```

Install `mpv` for local playback of the MPEG stream.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, ElevenlabsEngine


if __name__ == "__main__":
    engine = ElevenlabsEngine(voice="Nicole")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from ElevenLabs.")
    stream.play()
```

## Source Notes

- Constructor defaults include `voice="Nicole"`,
  `id="piTKgcLEGmPE4e6mEKli"`, `category="premade"`, and
  `model="eleven_multilingual_v2"`.
- `get_voices()` calls the ElevenLabs API and returns `ElevenlabsVoice`
  objects.
- The current source comments note that newer ElevenLabs stream endpoints do not
  accept clarity, stability, or style values per request. The constructor keeps
  those fields for compatibility.
- Output reports `pyaudio.paCustomFormat, -1, -1`, so compressed playback needs
  the external player path.

## Troubleshooting

- Missing key errors can be fixed by passing `api_key` or setting
  `ELEVENLABS_API_KEY`.
- If a voice name is not found, pass the explicit voice ID in the constructor.
