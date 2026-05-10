# System Engine

`SystemEngine` wraps the operating system TTS voices through `pyttsx3`. It is
the best first smoke test because it does not need cloud credentials or model
downloads.

## Install

```bash
pip install "realtimetts[system]"
```

This installs `pyttsx3` plus the RealtimeTTS playback dependencies. Available
voices depend entirely on the host OS.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, SystemEngine


if __name__ == "__main__":
    stream = TextToAudioStream(SystemEngine(voice="Zira"))
    stream.feed("Hello from the local system engine.")
    stream.play()
```

## Source Notes

- `SystemVoice(name, id)` stores the OS voice name and ID.
- `SystemEngine(voice="Zira", print_installed_voices=False)` selects the first
  installed voice whose name contains the requested string.
- `set_voice_parameters(**kwargs)` forwards values directly to `pyttsx3`
  `setProperty`, so rate, volume, and other support are platform-dependent.
- Output is queued as mono 16-bit PCM at 22050 Hz.

## Troubleshooting

- If voice selection appears to do nothing, call
  `SystemEngine(print_installed_voices=True)` and use a substring from an
  installed voice name.
- On Linux, install PortAudio headers before installing PyAudio. Some systems
  also need an OS speech backend configured for `pyttsx3`.
