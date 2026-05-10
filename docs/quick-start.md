# Quick Start

This page gets one local engine speaking first, then shows the streaming shape
used by LLM applications.

## Install A Starter Engine

```bash
pip install "realtimetts[system]"
```

On Linux, install PortAudio headers before installing PyAudio:

```bash
sudo apt-get update
sudo apt-get install python3-dev portaudio19-dev
```

On macOS:

```bash
brew install portaudio
```

See [installation](installation.md) for optional engine extras, API keys, local
model setup, `mpv`, CUDA, and packaging caveats.

## First Audio

```python
from RealtimeTTS import TextToAudioStream, SystemEngine


if __name__ == "__main__":
    engine = SystemEngine()
    stream = TextToAudioStream(engine)
    stream.feed("Hello from RealtimeTTS.")
    stream.play()
```

Use the `if __name__ == "__main__":` guard in scripts, especially on Windows and
when using engines that start worker processes.

## Feed A Generator

`feed()` accepts strings and iterators. This is the same shape used for LLM
streaming, where each yielded item is a text chunk.

```python
from RealtimeTTS import TextToAudioStream, SystemEngine


def text_chunks():
    yield "This starts speaking quickly. "
    yield "More text can arrive while audio is already playing."


if __name__ == "__main__":
    stream = TextToAudioStream(SystemEngine())
    stream.feed(text_chunks())
    stream.play()
```

## Play In The Background

```python
import time
from RealtimeTTS import TextToAudioStream, SystemEngine


if __name__ == "__main__":
    stream = TextToAudioStream(SystemEngine())
    stream.feed("This plays asynchronously.")
    stream.play_async()

    while stream.is_playing():
        time.sleep(0.1)
```

`play()` blocks until playback finishes. `play_async()` starts playback in a
background thread and returns immediately.

## Save Audio Without Local Playback

```python
from RealtimeTTS import TextToAudioStream, SystemEngine


if __name__ == "__main__":
    stream = TextToAudioStream(SystemEngine())
    stream.feed("Write this speech to a WAV file.")
    stream.play(output_wavfile="speech.wav", muted=True)
```

See [output and files](output-and-files.md) for audio chunks, output devices,
and engines that use `mpv` for compressed audio.

## Next Steps

- Pick an engine in [engine selection](engine-selection.md).
- Learn the playback lifecycle in [feed and playback](feed-and-playback.md).
- Connect streamed model output in [LLM streaming](llm-streaming.md).
