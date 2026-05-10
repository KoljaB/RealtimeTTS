# RealtimeTTS

[![PyPI](https://img.shields.io/pypi/v/RealtimeTTS)](https://pypi.org/project/RealtimeTTS/)
[![Downloads](https://static.pepy.tech/badge/RealtimeTTS)](https://www.pepy.tech/projects/realtimetts)
[![GitHub release](https://img.shields.io/github/release/KoljaB/RealtimeTTS.svg)](https://github.com/KoljaB/RealtimeTTS/releases/)

RealtimeTTS is a Python text-to-speech library for applications that need to
turn strings, generators, and LLM token streams into audio with low latency. It
can play speech locally, stream chunks to another process, write WAV files, and
fall back across multiple engines.

The project supports a broad engine matrix: local system voices, cloud APIs,
free service wrappers, local neural models, and voice-cloning stacks.

## Project Status

RealtimeTTS is community-driven. Focused pull requests are welcome, while
maintainer availability for new features and support may be limited.

## Install

For the fastest local smoke test, install the system engine:

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

For cloud engines, local neural engines, CUDA, `mpv`, and current packaging
caveats, see [docs/installation.md](docs/installation.md).

## First Audio

```python
from RealtimeTTS import TextToAudioStream, SystemEngine


if __name__ == "__main__":
    stream = TextToAudioStream(SystemEngine())
    stream.feed("Hello from RealtimeTTS.")
    stream.play()
```

Use the `if __name__ == "__main__":` guard in scripts, especially on Windows and
when using engines that start worker processes.

## Streaming Text

`feed()` accepts an iterator, so text can arrive while audio is already playing:

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

Use the same pattern with an LLM client by yielding only non-empty text chunks.
See [docs/llm-streaming.md](docs/llm-streaming.md).

## Output

Write audio to a WAV file without local speaker playback:

```python
from RealtimeTTS import TextToAudioStream, SystemEngine


if __name__ == "__main__":
    stream = TextToAudioStream(SystemEngine())
    stream.feed("Save this speech to a file.")
    stream.play(output_wavfile="speech.wav", muted=True)
```

For output devices, `mpv` playback, muted mode, callbacks, and chunk formats,
see [docs/output-and-files.md](docs/output-and-files.md).

## Features

- Low-latency playback from strings, generators, and streamed model output.
- Multiple engines with local, cloud, free-service, and neural model options.
- Fallback engines for more resilient synthesis.
- Sync and async playback with pause, resume, stop, and state inspection.
- Text, audio, sentence, character, word-timing, and audio-chunk callbacks.
- WAV output, muted synthesis, selected output devices, and volume control.
- Voice switching and voice-cloning workflows where supported by the engine.

## Engine Overview

| Engine | Type | Install/status note | Best first use |
| --- | --- | --- | --- |
| [`SystemEngine`](docs/engines/system.md) | Local | `realtimetts[system]` | First local audio smoke test. |
| [`GTTSEngine`](docs/engines/gtts.md) | Free service | `realtimetts[gtts]` | Simple network-backed speech. |
| [`EdgeEngine`](docs/engines/edge.md) | Free service | `realtimetts[edge]`, needs `mpv` | Free streamed voices. |
| [`OpenAIEngine`](docs/engines/openai.md) | Cloud API | `realtimetts[openai]` | OpenAI TTS voices. |
| [`AzureEngine`](docs/engines/azure.md) | Cloud API | `realtimetts[azure]` | Azure voices and word timings. |
| [`ElevenlabsEngine`](docs/engines/elevenlabs.md) | Cloud API | `realtimetts[elevenlabs]`, needs `mpv` | High-quality API voices. |
| [`CambEngine`](docs/engines/camb.md) | Cloud API | `realtimetts[camb]` | CAMB MARS API voices. |
| [`MiniMaxEngine`](docs/engines/minimax.md) | Cloud API | `realtimetts[minimax]` | MiniMax cloud voices. |
| [`CartesiaEngine`](docs/engines/cartesia.md) | Cloud API | `realtimetts[cartesia]` | Cartesia API voices. |
| [`TypecastEngine`](docs/engines/typecast.md) | Cloud API | `realtimetts[typecast]` | Typecast API voices. |
| [`ModelsLabEngine`](docs/engines/modelslab.md) | Cloud API | `realtimetts[modelslab]`, root export pending | ModelsLab API voices. |
| [`CoquiEngine`](docs/engines/coqui.md) | Local neural | `realtimetts[coqui]` | Local XTTS voice cloning. |
| [`PiperEngine`](docs/engines/piper.md) | Local executable | `realtimetts[piper]`, external Piper setup | Fast local executable TTS. |
| [`StyleTTSEngine`](docs/engines/styletts.md) | Local neural | `realtimetts[styletts]`, local checkout/assets | StyleTTS experiments. |
| [`ParlerEngine`](docs/engines/parler.md) | Local neural | `realtimetts[parler]` | GPU local model experiments. |
| [`KokoroEngine`](docs/engines/kokoro.md) | Local neural | `realtimetts[kokoro]` | Local voices and timing support. |
| [`OrpheusEngine`](docs/engines/orpheus.md) | Local/API-style | `realtimetts[orpheus]` | Orpheus model workflows. |
| [`FasterQwenEngine`](docs/engines/faster-qwen.md) | Local neural | `realtimetts[qwen]` | Qwen voice cloning. |
| [`OmniVoiceEngine`](docs/engines/omnivoice.md) | Local neural | `realtimetts[omnivoice]` | Multilingual voice cloning. |
| [`PocketTTSEngine`](docs/engines/pockettts.md) | Local lightweight | `realtimetts[pockettts]` | CPU-oriented voice cloning. |
| [`NeuTTSEngine`](docs/engines/neutts.md) | Local neural | `realtimetts[neutts]`, optional `neutts-gguf` | Reference-audio voice cloning. |
| [`ZipVoiceEngine`](docs/engines/zipvoice.md) | Local neural | `realtimetts[zipvoice]`, external checkout | ZipVoice cloning/server demos. |
| [`LuxTTSEngine`](docs/engines/luxtts.md) | Local neural | `realtimetts[luxtts]` | LuxTTS voice cloning. |
| [`ChatterboxEngine`](docs/engines/chatterbox.md) | Local neural | `realtimetts[chatterbox]` | Chatterbox prompt-audio voices. |
| [`SoproTTSEngine`](docs/engines/sopro.md) | Local neural | `realtimetts[sopro]` | Sopro reference-audio voices. |
| [`SopranoEngine`](docs/engines/soprano.md) | Local neural | `realtimetts[soprano]` | Soprano local synthesis. |
| [`MossTTSEngine`](docs/engines/moss-tts.md) | Local neural | `realtimetts[moss]`, runtime assets | MOSS-TTS experiments. |

See [docs/engine-selection.md](docs/engine-selection.md) before choosing an
engine for an application. The engine-specific docs are being split out from the
old README and source audit.

## Documentation

- [Quick start](docs/quick-start.md): shortest working examples.
- [Installation](docs/installation.md): extras, platform setup, external tools,
  API keys, and known packaging mismatches.
- [Engine selection](docs/engine-selection.md): engine matrix and selection
  guidance.
- [Feed and playback](docs/feed-and-playback.md): `feed()`, `play()`,
  `play_async()`, pause, resume, stop, text state, and inline tags.
- [LLM streaming](docs/llm-streaming.md): provider-neutral streamed text
  patterns and latency tuning.
- [Output and files](docs/output-and-files.md): WAV files, audio chunks, muted
  mode, output devices, mpv, buffering, and volume.
- [Engine setup pages](docs/engine-selection.md) now link one focused page for
  each concrete engine source.
- [FAQ](FAQ.md): legacy troubleshooting page while topic docs are being split
  out.

Legacy translated docs remain under `docs/<locale>/` while English is refactored
as the canonical source.

## Server Example

The browser and WebSocket server example lives in `example_fast_api/`:

```bash
python -m pip install fastapi uvicorn websockets pyaudio
python example_fast_api/async_server.py
```

Open `http://localhost:8000` or connect to `ws://localhost:8000/ws`.

## Related Project

[RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) is the speech-to-text
counterpart for realtime voice input.

## Contributing

Focused docs, tests, and engine fixes are easiest to review. During the docs
refactor, keep English docs canonical and note mismatches between source,
packaging, examples, and tests rather than hiding them.

## License

RealtimeTTS source code is MIT licensed. Engine providers, model weights, voice
data, datasets, generated audio, and third-party services can have separate
terms. Read [LICENSING_ADDENDUM.md](LICENSING_ADDENDUM.md) and the relevant
provider or model licenses before commercial use.

Audio samples derived from the EARS dataset by Meta are licensed under CC BY-NC
4.0. See the original dataset terms for details.

## Author

Kolja Beigel
