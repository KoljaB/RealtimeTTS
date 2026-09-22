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

### Support RealtimeTTS

If RealtimeTTS saved you time, one GitHub star is a simple way to help make it
more stable.

Stars improve visibility, and visibility brings more users, more real-world
testing, more bug reports, more fixes, and better releases for everyone.

## Demo

https://github.com/KoljaB/RealtimeTTS/assets/7604638/87dcd9a5-3a4e-4f57-be45-837fc63237e7

## Recommended Engine: Qwen (GPU and CPU)

For supported Windows and Linux systems with an NVIDIA GPU, **QwenEngine is
currently the recommended and preferred RealtimeTTS engine for high-quality,
low-latency conversational speech**. It offers multilingual Qwen3-TTS quality,
x-vector and ICL voice cloning, native 24 kHz PCM streaming, fast cancellation,
and the same engine either in-process or behind the production Qwen server.

In 10 warm Linux runs on our tuned RTX 4090 setup, the timeline was: about
**35 ms engine TTFT**, another **35 ms until RealtimeTTS emits its first PCM
chunk**, and about **10 ms of silence inside that chunk**. Predicted audible
onset was **80.9 ms** and RTF was **0.108**. These are orientation figures;
measure the complete path on your target system.

RealtimeTTS 0.8.6 also provides `QwenCpuEngine`, using the maintained CPU-only
native runtime with worker-pool improvements and onset recovery. Choose one of
these server installations in a fresh Python 3.11 or 3.12 virtual environment:

```bash
# NVIDIA GPU on Windows or Linux x86-64
python -m pip install "realtimetts[qwen-server]"
realtimetts-qwen-server --clone-mode speaker_only
```

```bash
# CPU on Windows, Linux, Intel Mac, or Apple Silicon
python -m pip install "realtimetts[qwen-cpu-server]"
realtimetts-qwen-server --device cpu --clone-mode speaker_only --no-clamp-fp16 --onset-silence-profile qwen3_tts_12hz_0_6b_base_q8_v1 --onset-silence-recovery
```

Neither server extra needs Torch, a local CUDA Toolkit, or PortAudio. GPU mode
still needs a compatible NVIDIA GPU/driver. CPU wheels support Windows x86-64,
Linux x86-64 with glibc 2.35+, Intel macOS 13+, and Apple Silicon macOS 11+.
x86-64 CPUs require AVX2/FMA/F16C/BMI2. CPU throughput depends on the machine;
older x86 CPUs, Linux ARM CPU, Windows ARM64, and Metal are outside this release.

Both servers offer browser playback at `http://127.0.0.1:8080/studio`, early
em-dash speech, streaming segment controls, and language detection. See the
[QwenEngine guide](docs/engines/qwen.md) for models, voices, authentication,
language routing, and reproducing the deployed CPU settings.

CPU decoder overlap is available through explicit worker and chunk settings.
See [CPU scheduling and measured results](docs/qwen-cpu-scheduling.md).

### Emotional Qwen demo from the video

The complete editable demo is in `tests/faster_qwen_emotions.py`, including the
voice texts and playback loop. Its compact colored output is the default;
add `--verbose` for native diagnostics. Warnings and errors remain visible.
The packaged `RealtimeTTS.qwen_emotions` command is built from the same source.
It keeps the original eleven emotional references/texts and the 0.6B Base Q8
speaker-only workflow. In an activated Python 3.11 or 3.12 environment:

```bash
git clone https://github.com/KoljaB/RealtimeTTS.git
cd RealtimeTTS
python -m pip install -e ".[qwen]"
cd tests
python faster_qwen_emotions.py
```

For CPU use `.[qwen-cpu]` when installing, then run
`python faster_qwen_emotions.py --device cpu`. Local playback on Linux/macOS
needs PortAudio (see below). The first run downloads the model pair if needed.
See the [emotional showcase guide](docs/qwen-emotions.md) for the complete
virtual-environment setup, headless/server mode, and package-only commands.

[`InflectEngine`](docs/engines/inflect.md) is the documented lightweight
alternative for one fixed English voice on CUDA or ONNX CPU.

## Install

For the fastest local smoke test, install the system engine:

```bash
pip install "realtimetts[system]"
```

The `system` and other traditional engine extras use PyAudio. On Linux, install
PortAudio headers before those extras:

```bash
sudo apt-get update
sudo apt-get install python3-dev portaudio19-dev
```

On macOS:

```bash
brew install portaudio
```

The local `qwen`, `qwen-cpu`, and Inflect extras use PyAudio/PortAudio for
playback. Windows has prebuilt PyAudio wheels; on Linux and macOS install
PortAudio first using the commands above. Use Python 3.11 or 3.12 for the
Qwen workflows. Server-only installations do not need audio-device libraries.

Install `realtimetts[qwen-server]` (GPU) or `realtimetts[qwen-cpu-server]` (CPU)
to expose the same native engine through
an OpenAI-compatible HTTP API. The server provides `/v1/audio/speech`, dynamic
voice registration, persistent voice latents, and watchdog-ready request/stall
metrics on `/health`; it is headless and does not install PyAudio/PortAudio.
The server defaults to loopback (`127.0.0.1`). LAN exposure requires a
deliberate `--allow-lan` bind plus a built-in API key, or a trusted reverse
proxy that terminates TLS and enforces authentication. CORS defaults to
explicit localhost origins and rejects wildcard `*`; CORS is not an access
control boundary. See [the Qwen guide](docs/engines/qwen.md#http-server) for
deployment, protocol, licensing, and asset boundaries.

Sentence splitting defaults to stream2sentence's `nltk+rule-based` consensus
mode. The normal install, including `realtimetts[qwen]`, installs
`stream2sentence[nltk]` but not Stanza or PyTorch. Add Stanza only when wanted:

```bash
pip install "realtimetts[stanza]"
# or
pip install "realtimetts[qwen,stanza]"
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
| **[`QwenEngine`](docs/engines/qwen.md) (recommended)** | Local native neural / HTTP server | `realtimetts[qwen]` or `realtimetts[qwen-server]` with a matching native wheel | High-quality multilingual realtime speech, voice cloning, and fast cancellation. |
| [`QwenCpuEngine`](docs/engines/qwen.md#cpu-engine) | Local CPU native / HTTP server | `realtimetts[qwen-cpu]` or `realtimetts[qwen-cpu-server]` | CPU-only Qwen on Windows/Linux x86-64 and Intel/Apple Silicon macOS; onset recovery and the same streaming controls. |
| [`InflectEngine`](docs/engines/inflect.md) | Local lightweight | `realtimetts[inflect]` | Fast fixed English voice through PyTorch CUDA or ONNX CPU. |
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
| [`ModelsLabEngine`](docs/engines/modelslab.md) | Cloud API | `realtimetts[modelslab]` | ModelsLab API voices. |
| [`CoquiEngine`](docs/engines/coqui.md) | Local neural | `realtimetts[coqui]` | Local XTTS voice cloning. |
| [`PiperEngine`](docs/engines/piper.md) | Local executable | `realtimetts[piper]`, external Piper setup | Fast local executable TTS. |
| [`StyleTTSEngine`](docs/engines/styletts.md) | Local neural | `realtimetts[styletts]`, local checkout/assets | StyleTTS experiments. |
| [`ParlerEngine`](docs/engines/parler.md) | Local neural | `realtimetts[parler]` | GPU local model experiments. |
| [`KokoroEngine`](docs/engines/kokoro.md) | Local neural | `realtimetts[kokoro]` | Local voices and timing support. |
| [`OrpheusEngine`](docs/engines/orpheus.md) | Local/API-style | `realtimetts[orpheus]` | Orpheus model workflows. |
| [`OmniVoiceEngine`](docs/engines/omnivoice.md) | Local neural | `realtimetts[omnivoice]` | Multilingual voice cloning. |
| [`PocketTTSEngine`](docs/engines/pockettts.md) / `PocketTTSGpuEngine` | Local lightweight | `realtimetts[pockettts]`, `realtimetts[pockettts-gpu]` plus GPU fork | CPU-oriented voice cloning, optional CUDA fork path. |
| [`NeuTTSEngine`](docs/engines/neutts.md) | Local neural | `realtimetts[neutts]`, optional `neutts-gguf` | Reference-audio voice cloning. |
| [`ZipVoiceEngine`](docs/engines/zipvoice.md) | Local neural | `realtimetts[zipvoice]`, external checkout | ZipVoice cloning/server demos. |
| [`LuxTTSEngine`](docs/engines/luxtts.md) | Local neural | `realtimetts[luxtts]` | LuxTTS voice cloning. |
| [`ChatterboxEngine`](docs/engines/chatterbox.md) | Local neural | `realtimetts[chatterbox]` | Chatterbox prompt-audio voices. |
| [`SoproTTSEngine`](docs/engines/sopro.md) | Local neural | `realtimetts[sopro]` | Sopro reference-audio voices. |
| [`SopranoEngine`](docs/engines/soprano.md) | Local neural | `realtimetts[soprano]` | Soprano local synthesis. |
| [`MossTTSEngine`](docs/engines/moss-tts.md) | Local neural | `realtimetts[moss]`, runtime assets | MOSS-TTS experiments. |
| [`HiggsEngine`](docs/engines/higgs.md) | Local HTTP PCM server | `realtimetts[higgs]`, separate SGLang-Omni server | Stream Higgs Audio v3 PCM from a trusted server. |

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
- [Forced alignment](docs/ctc-forced-alignment.md): optional word and character
  timing for engines without native timings.
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

For the native Qwen/Inflect paths, `qwentts.cpp` and `realtimetts-qwen-native` are
MIT-licensed; Qwen 0.6B Base/tokenizer and Inflect Micro-v2/ONNX are
Apache-2.0. Model weights, voice latents, and reference audio are not bundled,
and users are responsible for the rights to every voice or recording they
provide.

Audio samples derived from the EARS dataset by Meta are licensed under CC BY-NC
4.0. See the original dataset terms for details.

## Author

Kolja Beigel
