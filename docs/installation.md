# Installation

RealtimeTTS installs a small core plus optional engine dependencies. The safest
first install is one engine extra that matches the engine you plan to use.

## Starter Install

```bash
pip install "realtimetts[system]"
```

The `system` extra installs the local system TTS path through `pyttsx3`. It is
usually the quickest way to test that Python, PyAudio, and audio output are
working.

## Platform Audio Prerequisites

RealtimeTTS uses PyAudio/PortAudio for supported local PCM playback, including
the native `qwen` and Inflect extras. Python 3.13+ also installs the small
`audioop-lts` compatibility wheel required by pydub; this does not compile
locally.

Linux:

```bash
sudo apt-get update
sudo apt-get install python3-dev portaudio19-dev
```

macOS:

```bash
brew install portaudio
```

Windows installs PyAudio wheels directly on Python 3.10–3.13. PyAudio 0.2.14
does not publish a Windows Python 3.14 wheel, so use Python 3.13 for supported
RealtimeTTS local playback.

To add PyAudio explicitly, use `realtimetts[playback]`.

## Sentence Tokenizer

RealtimeTTS defaults to stream2sentence's `nltk+rule-based` consensus mode.
The normal install, including `realtimetts[qwen]`, therefore forwards the
`stream2sentence[nltk]` extra but does not install Stanza or its PyTorch stack.
NLTK downloads its small tokenizer data (currently `punkt_tab`, and `punkt`
where required by the installed NLTK version) on first use when it is not
already cached.

Stanza is opt-in:

```bash
pip install "realtimetts[stanza]"
# or together with an engine:
pip install "realtimetts[qwen,stanza]"
```

Select `tokenizer="rule-based"` when an application intentionally wants to use
only the local boundary rules at runtime. This selection does not remove NLTK
from an already installed RealtimeTTS environment. The `nltk` extra is also
exposed as an explicit, idempotent alias for deployment manifests, although
NLTK is already part of the default installation.

## Current Extras

These extras are present in `setup.py`:

| Extra | Intended use |
| --- | --- |
| `minimal` | Core streaming dependencies only. |
| `playback` | Traditional PyAudio playback backend. |
| `nltk` | Default NLTK plus rule-based consensus tokenizer; included by default. |
| `stanza` | Optional Stanza tokenizer and its runtime dependencies. |
| `system` | System TTS through `pyttsx3`. |
| `azure` | Azure Speech SDK. |
| `elevenlabs` | ElevenLabs SDK. |
| `openai` | OpenAI SDK. |
| `gtts` | Google Text-to-Speech package. |
| `coqui` | Coqui TTS package. |
| `edge` | Microsoft Edge TTS package. |
| `kokoro` | Kokoro engine package. |
| `camb` | CAMB SDK. |
| `minimax` | MiniMax engine dependencies. |
| `modelslab` | ModelsLab engine dependencies. |
| `cartesia` | Cartesia SDK. |
| `typecast` | Typecast SDK. |
| `orpheus` | SNAC dependency used by Orpheus. |
| `omnivoice` | OmniVoice package. |
| `luxtts` | LuxTTS-related Git dependencies and local stack packages. |
| `zipvoice` | Shared ZipVoice Python dependencies; still needs a ZipVoice checkout. |
| `chatterbox` | Chatterbox TTS package. |
| `inflect` | Inflect-Micro-v2 PyTorch and ONNX runtimes. |
| `inflect-pytorch`, `inflect-onnx` | Backend-specific Inflect dependencies; the ONNX engine path itself does not use Torch. |
| `sopro` | Sopro package. |
| `soprano` | Soprano TTS package. |
| `neutts`, `neutts-gguf` | NeuTTS package and optional GGUF/ONNX extras. |
| `pockettts`, `pocket` | PocketTTS package. |
| `pockettts-gpu`, `pocket-gpu` | Shared dependencies for `PocketTTSGpuEngine`; install a CUDA PyTorch build and the pinned PocketTTS GPU fork separately. |
| `styletts`, `style` | StyleTTS Python dependencies; still needs a StyleTTS checkout/assets. |
| `parler` | PyPI-resolvable Parler support dependencies; install the upstream Parler package separately. |
| `moss`, `moss-tts` | PyPI-resolvable MOSS runtime dependencies; install MOSS-TTS-Nano/model assets separately. |
| `piper` | Core RealtimeTTS dependencies; Piper binary/model assets remain external. |
| `qwen` | Native in-process qwentts.cpp backend with PyAudio playback (the `0.7.4.dev9` candidate selects Windows `0.4.0.dev1` or Linux `0.4.0.dev0`). |
| `qwen-server` | OpenAI-compatible HTTP server for the native Qwen backend (same platform-specific native wheel pins, without PyAudio). |
| `inflect`, `inflect-pytorch`, `inflect-onnx` | Inflect-Micro-v2 with PyAudio playback; choose both backends, PyTorch only, or ONNX only. |
| `jp`, `zh`, `ko` | Extra language support packages for Kokoro. |
| `all` | Best-effort convenience set for all Python-installable engine stacks. |

Example:

```bash
pip install "realtimetts[azure,openai]"
```

## Engine-Specific Setup

The table below is intentionally more explicit than the README. Some engines are
covered by package extras, while others currently require an upstream package,
local checkout, model files, or Docker example.

| Engine | Install or setup path | Extra setup |
| --- | --- | --- |
| [`SystemEngine`](engines/system.md) | `pip install "realtimetts[system]"` | Uses system voices through `pyttsx3`. |
| [`GTTSEngine`](engines/gtts.md) | `pip install "realtimetts[gtts]"` | Network access to the Google Translate TTS service. |
| [`EdgeEngine`](engines/edge.md) | `pip install "realtimetts[edge]"` | Install `mpv` for compressed audio playback. |
| [`OpenAIEngine`](engines/openai.md) | `pip install "realtimetts[openai]"` | Set `OPENAI_API_KEY`; install `mpv` for MP3 playback or use PCM response format. |
| [`AzureEngine`](engines/azure.md) | `pip install "realtimetts[azure]"` | Pass `speech_key` and `service_region`; source does not currently read Azure env vars. |
| [`ElevenlabsEngine`](engines/elevenlabs.md) | `pip install "realtimetts[elevenlabs]"` | Set `ELEVENLABS_API_KEY`; install `mpv`. |
| [`CambEngine`](engines/camb.md) | `pip install "realtimetts[camb]"` | Set `CAMB_API_KEY`. |
| [`MiniMaxEngine`](engines/minimax.md) | `pip install "realtimetts[minimax]"` | Set `MINIMAX_API_KEY`; install `mpv` for MP3 playback. |
| [`CartesiaEngine`](engines/cartesia.md) | `pip install "realtimetts[cartesia]"` | Set `CARTESIA_API_KEY`. |
| [`TypecastEngine`](engines/typecast.md) | `pip install "realtimetts[typecast]"` | Set `TYPECAST_API_KEY` and provide `voice_id` or `TYPECAST_VOICE_ID`. |
| [`ModelsLabEngine`](engines/modelslab.md) | `pip install "realtimetts[modelslab]"` | Set `MODELSLAB_API_KEY`; import from `RealtimeTTS` or `RealtimeTTS.engines`. |
| [`CoquiEngine`](engines/coqui.md) | `pip install "realtimetts[coqui]"` | Local XTTS model download/cache; GPU strongly recommended for realtime use. |
| [`PiperEngine`](engines/piper.md) | `pip install "realtimetts[piper]"` plus Piper executable/model files. | Provide a Piper executable, model, and config; `PIPER_PATH` can point to the executable. |
| [`StyleTTSEngine`](engines/styletts.md) | `pip install "realtimetts[styletts]"` plus StyleTTS2 checkout/model files. | Pass `style_root`, model config, checkpoint, and reference audio. |
| [`ParlerEngine`](engines/parler.md) | `pip install "realtimetts[parler]"` plus the upstream Parler package. | Torch/torchaudio and GPU setup are usually required for realtime performance. |
| [`KokoroEngine`](engines/kokoro.md) | `pip install "realtimetts[kokoro]"` | Add `jp`, `zh`, or `ko` extras for those language stacks. |
| [`OrpheusEngine`](engines/orpheus.md) | `pip install "realtimetts[orpheus]"` | Requires an OpenAI-compatible completions endpoint such as a local LM Studio server. |
| [`QwenEngine`](engines/qwen.md) | `pip install "realtimetts[qwen]"` | Matching native wheel and NVIDIA CUDA-12 driver; no local CUDA Toolkit. Use the headless `realtimetts[qwen-server]` extra for the HTTP server without PyAudio. |
| [`OmniVoiceEngine`](engines/omnivoice.md) | `pip install "realtimetts[omnivoice]"` | Requires reference audio and exact reference text. |
| [`PocketTTSEngine`](engines/pockettts.md) / `PocketTTSGpuEngine` | `pip install "realtimetts[pockettts]"`; for GPU use `pip install "realtimetts[pockettts-gpu]"`, install CUDA PyTorch, then install the pinned PocketTTS GPU fork. | Optional prompt WAV for voice cloning; CPU-oriented default, separate CUDA fork engine. |
| [`NeuTTSEngine`](engines/neutts.md) | `pip install "realtimetts[neutts]"`; use `realtimetts[neutts-gguf]` for NeuTTS optional extras. | Use `neutts[llama,onnx]` and GGUF for low-latency streaming. |
| [`ZipVoiceEngine`](engines/zipvoice.md) | `pip install "realtimetts[zipvoice]"` plus a ZipVoice checkout passed as `zipvoice_root`. | Needs prompt WAV and exact transcript; use distill with at least 3 steps for fast quality work. |
| [`LuxTTSEngine`](engines/luxtts.md) | `pip install "realtimetts[luxtts]"` or install LuxTTS separately. | Pass `lux_root` if using a local LuxTTS checkout; requires prompt WAV/text. |
| [`ChatterboxEngine`](engines/chatterbox.md) | `pip install "realtimetts[chatterbox]"` | Uses `chatterbox-tts`; prompt WAV should be longer than 5 seconds. |
| [`InflectEngine`](engines/inflect.md) | `pip install "realtimetts[inflect]"` | Downloads a pinned Micro-v2 snapshot; supports PyTorch CUDA and ONNX CPU. |
| [`SoproTTSEngine`](engines/sopro.md) | `pip install "realtimetts[sopro]"` | Uses `sopro`; optional Hugging Face cache/token and reference WAV. |
| [`SopranoEngine`](engines/soprano.md) | `pip install "realtimetts[soprano]"` | Uses `soprano-tts`; single-voice English, no cloning. |
| [`MossTTSEngine`](engines/moss-tts.md) | `pip install "realtimetts[moss]"` or install MOSS-TTS-Nano separately. | Needs MOSS model/runtime assets; ONNX and torch backends have different dependencies. |

## Cloud Credentials

Cloud engines usually accept an API key constructor argument and also read an
environment variable. Azure is the current exception: older docs mention env
vars, but the source constructor takes direct key and region arguments.

| Engine | Credential path observed |
| --- | --- |
| OpenAI | `OPENAI_API_KEY` |
| Azure | Constructor arguments `speech_key` and `service_region` |
| ElevenLabs | `ELEVENLABS_API_KEY` |
| CAMB | `CAMB_API_KEY` |
| MiniMax | `MINIMAX_API_KEY` |
| Cartesia | `CARTESIA_API_KEY` |
| Typecast | `TYPECAST_API_KEY`, optional `TYPECAST_VOICE_ID` |
| ModelsLab | `MODELSLAB_API_KEY` |

## External Tools

Some engines need tools or assets outside Python packages.

| Requirement | Used by | Notes |
| --- | --- | --- |
| `mpv` | Engines that stream compressed audio, including Edge, ElevenLabs, OpenAI MP3, MiniMax, and ModelsLab. | Run `mpv --audio-device=help` to inspect mpv output device names. |
| `ffmpeg` | Audio conversion workflows through `pydub`. | Install from your OS package manager or ffmpeg.org. |
| Piper executable and model files | `PiperEngine` | `PIPER_PATH` can point to the executable. |
| Local model checkouts or Hugging Face assets | Many local neural engines | Needed by engines such as Coqui, Parler, StyleTTS2, ZipVoice, LuxTTS, Sopro, Soprano, and MOSS-TTS. |
| CUDA, PyTorch, torchaudio, CUDNN | Local neural engines | Exact requirements vary by engine and model. |
| NLTK `punkt` and `punkt_tab` data | Sentence splitting around many neural engine tests | Several Zaphod venvs needed local tokenizer data to avoid blocked online lookups. |

## Packaging Notes

Some engines need setup outside the Python extras, and a few compatibility notes
are worth checking before choosing an engine:

- Engine classes are lazily imported from both `RealtimeTTS` and
  `RealtimeTTS.engines`; optional engine dependencies are loaded only when the
  corresponding class is first accessed.
- `PiperEngine` still needs an external executable and voice model; its setup
  extra cannot install those assets.
- `StyleTTSEngine` and `ZipVoiceEngine` still need local upstream checkouts and
  model assets even though setup extras now install their Python dependency
  scaffolding.
- `[all]` is now broader, but it is a best-effort Python dependency set and
  still cannot install OS tools, CUDA builds, local model files, or provider
  accounts.
- `setup.py` declares Python `>=3.9, <3.15`, while older docs still say
  `<3.13`.

### Build and validate a release

Build both release artifacts from a clean checkout with the PEP 517 frontend,
then validate the metadata before publishing:

```bash
python -m pip install --upgrade build twine
python -m build --sdist --wheel
python -m twine check --strict dist/*
python tools/clean_install_smoke.py dist
```

The clean-install smoke creates a disposable virtual environment, installs the
wheel with `--no-deps --no-index`, checks that the installed distribution
version matches `RealtimeTTS.__version__`, and confirms the Qwen server entry
point is present. It does not download models or optional engine dependencies.
Run the focused unit suite separately:

```bash
python -m pip install pytest numpy requests
python -m pytest -q \
  tests/test_base_engine_silence_trim.py \
  tests/test_inflect_engine.py \
  tests/test_language_router.py \
  tests/test_minimax_engine.py \
  tests/test_release_metadata.py
```

### TestPyPI candidate installs

Keep dependency resolution on the normal PyPI index while taking the two
coordinated candidate wheels from TestPyPI. Install the native candidate with
`--no-deps` first so pip cannot silently select an older incompatible ABI. The
final command uses the TestPyPI project page only as a find-links source for
the exact RealtimeTTS candidate:

Choose exactly one native-candidate command for the host platform.

Windows x86-64:

```bash
python -m pip install --no-deps \
  --index-url https://test.pypi.org/simple \
  "qwentts-cpp-python[cuda12]==0.4.0.dev1"
```

Linux x86-64:

```bash
python -m pip install --no-deps \
  --index-url https://test.pypi.org/simple \
  "qwentts-cpp-python[cuda12]==0.4.0.dev0"
```

Then install the shared runtime dependencies and the headless server extra:

```bash
python -m pip install \
  --index-url https://pypi.org/simple \
  "numpy" "huggingface-hub" \
  "nvidia-cuda-runtime-cu12>=12.8,<13" "nvidia-cublas-cu12>=12.8,<13"
python -m pip install \
  --index-url https://pypi.org/simple \
  --find-links https://test.pypi.org/simple/realtimetts/ \
  "realtimetts[qwen-server]==0.7.4.dev9"
python -m qwentts_cpp doctor
python -m pip check
```

For a non-Qwen candidate, replace the first command with the PyPI extra you
need and keep the final command pinned to the exact TestPyPI version. Never use
TestPyPI as the sole index for an install that resolves transitive dependencies.

See [the source inventory](refactor-source-inventory.md) for the full audit
notes.
