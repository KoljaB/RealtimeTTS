# MiniMax Engine

`MiniMaxEngine` uses the MiniMax T2A v2 cloud API. It returns MP3 audio and
requires a MiniMax API key.

## Install

```bash
pip install "realtimetts[minimax]"
```

Set the API key in the environment or pass it to the constructor:

```powershell
$env:MINIMAX_API_KEY = "..."
```

Install `mpv` for local playback of MP3 output.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, MiniMaxEngine


if __name__ == "__main__":
    engine = MiniMaxEngine(model="speech-2.8-hd", voice="English_Graceful_Lady")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from MiniMax.")
    stream.play()
```

## Source Notes

- Model defaults to `speech-2.8-hd`; source comments also mention
  `speech-2.8-turbo` as a faster variant.
- Built-in voice presets include English and multilingual names such as
  `English_Graceful_Lady`, `English_Persuasive_Man`, `Wise_Woman`, and
  `Deep_Voice_Man`.
- `speed`, `volume`, and `pitch` are sent in `voice_setting`.
- Output is hex-encoded MP3 from the API and reports custom-format playback at
  32000 Hz.
- Unit and integration tests exist for MiniMax behavior.

## Troubleshooting

- API errors are logged with MiniMax status messages from `base_resp`.
- Playback issues usually mean the compressed audio path cannot find `mpv`.
