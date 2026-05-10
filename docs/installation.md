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

RealtimeTTS uses PyAudio for normal PCM playback.

Linux:

```bash
sudo apt-get update
sudo apt-get install python3-dev portaudio19-dev
```

macOS:

```bash
brew install portaudio
```

Windows usually installs PyAudio wheels directly. If PyAudio fails to install,
check that your Python version has a compatible wheel.

## Current Extras

These extras are present in `setup.py` after the first packaging alignment pass:

| Extra | Intended use |
| --- | --- |
| `minimal` | Core streaming dependencies only. |
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
| `sopro` | Sopro package. |
| `soprano` | Soprano TTS package. |
| `neutts`, `neutts-gguf` | NeuTTS package and optional GGUF/ONNX extras. |
| `pockettts`, `pocket` | PocketTTS package. |
| `styletts`, `style` | StyleTTS Python dependencies; still needs a StyleTTS checkout/assets. |
| `parler` | PyPI-resolvable Parler support dependencies; install the upstream Parler package separately. |
| `moss`, `moss-tts` | PyPI-resolvable MOSS runtime dependencies; install MOSS-TTS-Nano/model assets separately. |
| `piper` | Core RealtimeTTS dependencies; Piper binary/model assets remain external. |
| `qwen` | Faster Qwen3 TTS package. |
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
| [`ModelsLabEngine`](engines/modelslab.md) | `pip install "realtimetts[modelslab]"` | Set `MODELSLAB_API_KEY`; import from `RealtimeTTS.engines` until root export is added. |
| [`CoquiEngine`](engines/coqui.md) | `pip install "realtimetts[coqui]"` | Local XTTS model download/cache; GPU strongly recommended for realtime use. |
| [`PiperEngine`](engines/piper.md) | `pip install "realtimetts[piper]"` plus Piper executable/model files. | Provide a Piper executable, model, and config; `PIPER_PATH` can point to the executable. |
| [`StyleTTSEngine`](engines/styletts.md) | `pip install "realtimetts[styletts]"` plus StyleTTS2 checkout/model files. | Pass `style_root`, model config, checkpoint, and reference audio. |
| [`ParlerEngine`](engines/parler.md) | `pip install "realtimetts[parler]"` plus the upstream Parler package. | Torch/torchaudio and GPU setup are usually required for realtime performance. |
| [`KokoroEngine`](engines/kokoro.md) | `pip install "realtimetts[kokoro]"` | Add `jp`, `zh`, or `ko` extras for those language stacks. |
| [`OrpheusEngine`](engines/orpheus.md) | `pip install "realtimetts[orpheus]"` | Requires an OpenAI-compatible completions endpoint such as a local LM Studio server. |
| [`FasterQwenEngine`](engines/faster-qwen.md) | `pip install "realtimetts[qwen]"` | Needs reference audio/text or a speaker embedding; CUDA is the expected fast path. |
| [`OmniVoiceEngine`](engines/omnivoice.md) | `pip install "realtimetts[omnivoice]"` | Requires reference audio and exact reference text. |
| [`PocketTTSEngine`](engines/pockettts.md) | `pip install "realtimetts[pockettts]"` | Optional prompt WAV for voice cloning; CPU-oriented. |
| [`NeuTTSEngine`](engines/neutts.md) | `pip install "realtimetts[neutts]"`; use `realtimetts[neutts-gguf]` for NeuTTS optional extras. | Use `neutts[llama,onnx]` and GGUF for low-latency streaming. |
| [`ZipVoiceEngine`](engines/zipvoice.md) | `pip install "realtimetts[zipvoice]"` plus a ZipVoice checkout passed as `zipvoice_root`. | Needs prompt WAV and exact transcript; use distill with at least 3 steps for fast quality work. |
| [`LuxTTSEngine`](engines/luxtts.md) | `pip install "realtimetts[luxtts]"` or install LuxTTS separately. | Pass `lux_root` if using a local LuxTTS checkout; requires prompt WAV/text. |
| [`ChatterboxEngine`](engines/chatterbox.md) | `pip install "realtimetts[chatterbox]"` | Uses `chatterbox-tts`; prompt WAV should be longer than 5 seconds. |
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

## Known Packaging Mismatches

Do not treat the current extras as final release documentation yet. The Stage 0
inventory found mismatches that should be fixed or documented before release:

- `ModelsLabEngine` is exported from `RealtimeTTS.engines`, but not from the
  root `RealtimeTTS` lazy export table.
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

See [the source inventory](refactor-source-inventory.md) for the full audit
notes.
