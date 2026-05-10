# Documentation Source Inventory

This is the Stage 0 inventory for the documentation refactor described in
`roadmap/doc-refacturing.md`. It captures what the docs should treat as source
of truth before the README is reset.

Inventory date: 2026-05-10.

## Sources Inspected

| Area | Sources |
| --- | --- |
| Roadmap and style | `roadmap/doc-refacturing.md`, `D:\Projekte\STT\RealtimeSTT\RealtimeSTT\README.md` |
| Entry docs | `README.md`, `FAQ.md`, `LICENSE`, `LICENSING_ADDENDUM.md` |
| Docs site | `mkdocs.yml`, `docs/**` |
| Packaging | `setup.py`, `requirements.txt` |
| Public exports | `RealtimeTTS/__init__.py`, `RealtimeTTS/engines/__init__.py` |
| Stream API | `RealtimeTTS/text_to_stream.py`, `RealtimeTTS/stream_player.py` |
| Engines | `RealtimeTTS/engines/*_engine.py` |
| Examples and demos | `tests/*.py`, `example_fast_api/**`, `example_rvc/**`, `docker/**` |

## Current Docs Shape

| Item | Current state | README reset impact |
| --- | --- | --- |
| Root README | `README.md` is 836 lines and contains project intro, install notes, engine notes, examples, API reference, CUDA notes, testing, licensing, and history. | Reset should follow the compact RealtimeSTT style: short summary, working examples, engine overview, and docs map. |
| Root FAQ | `FAQ.md` is the real troubleshooting page. | Move symptom-based troubleshooting into canonical docs later; keep root FAQ stable during the first pass. |
| English docs | `docs/en/` has `index.md`, `installation.md`, `usage.md`, `api.md`, `contributing.md`, and `faq.md`. | These are thinner copies of README material and should not be the long-term English source. |
| Translations | `docs/` has 12 locale directories with the same 6-page shape: `ar`, `de`, `en`, `es`, `fr`, `hi`, `it`, `ja`, `ko`, `pt`, `ru`, `zh`. | Leave them untouched until the English canonical pages settle. |
| MkDocs nav | `mkdocs.yml` manually repeats each topic for each language. Language labels appear mojibaked and `ar` exists on disk but is not listed in `mkdocs.yml`. | Target nav should be topic-first, English-first, and should not manually duplicate every page per locale. |
| README language list | README lists `en`, `fr`, `es`, `de`, `it`, `zh`, `ja`, `hi`, and `ko`. | It misses `pt` and `ru` from MkDocs and `ar` from the docs tree. Do not preserve this list as-is in the reset. |

## Engine Inventory

Export status uses these meanings:

- `root`: exported from `RealtimeTTS/__init__.py`.
- `engines`: exported from `RealtimeTTS/engines/__init__.py`.

Each concrete engine source listed below now has a focused setup page under
`docs/engines/`. `base_engine.py` is intentionally excluded because it is the
engine base class, not an installable engine.

| Engine | Source | Export status | Setup extras observed | Examples/tests observed | Notes |
| --- | --- | --- | --- | --- | --- |
| `SystemEngine` | `system_engine.py` | root, engines | `system`, `all` | `tests/simple_test.py`, `tests/complex_test.py` | Low-friction README starter candidate. |
| `AzureEngine` | `azure_engine.py` | root, engines | `azure`, `all` | `tests/azure_test.py`, `tests/test_callbacks.py`, `tests/translator.py` | Uses Azure key and region; supports word timings. |
| `ElevenlabsEngine` | `elevenlabs_engine.py` | root, engines | `elevenlabs`, `all` | `tests/elevenlabs_test.py`, `tests/advanced_talk.py` | Uses MPEG/mpv playback path. |
| `CoquiEngine` | `coqui_engine.py` | root, engines | `coqui`, `all` | `tests/coqui_test.py`, `tests/coqui_test_stop.py`, `example_rvc/**` | Local neural voice cloning; needs shutdown guidance. |
| `OpenAIEngine` | `openai_engine.py` | root, engines | `openai`, `all` | `tests/openai_1.0_test.py`, `tests/openaitts_test.py`, `docker/openai-tts-docker/**` | README/docs should use current OpenAI examples only. |
| `GTTSEngine` | `gtts_engine.py` | root, engines | `gtts`, `all` | `tests/gtts_test.py` | Free service, no API key. |
| `ParlerEngine` | `parler_engine.py` | root, engines | `parler`, `all` | `tests/parler_test.py` | Extra keeps PyPI-resolvable support deps; upstream Parler package remains a manual Git install. |
| `EdgeEngine` | `edge_engine.py` | root, engines | `edge`, `all` | `tests/edge_test.py` | Uses MPEG/mpv playback path. |
| `StyleTTSEngine` | `style_engine.py` | root, engines | `styletts`, `style`, `all` | `tests/style_test.py` | Extra covers Python deps; local checkout/model setup is still required. |
| `PiperEngine` | `piper_engine.py` | root, engines | `piper`, `all` | `tests/piper_test.py` | Uses external Piper executable; `PIPER_PATH` is supported. |
| `KokoroEngine` | `kokoro_engine.py` | root, engines | `kokoro`, `jp`, `zh`, `ko`, `all` | `tests/kokoro_test.py`, `tests/kokoro_mix_voices.py` | Language extras and word-timing support need exact docs. |
| `OrpheusEngine` | `orpheus_engine.py` | root, engines | `orpheus`, `all` | `tests/orpheus_test.py` | Requires an OpenAI-compatible completions endpoint or local model stack. |
| `ZipVoiceEngine` | `zipvoice_engine.py` | root, engines | `zipvoice`, `all` | `tests/zipvoice_test.py`, `docker/zipvoice/**` | Extra covers shared deps; source checkout/assets still required. |
| `PocketTTSEngine` | `pocket_engine.py` | root, engines | `pockettts`, `pocket`, `all` | none focused found | Lazy import recommends `pip install pocket-tts`; docs need status and install caveat. |
| `NeuTTSEngine` | `neutts_engine.py` | root, engines | `neutts`, `neutts-gguf`, `all` | `tests/neutts_test.py` | GGUF/ONNX fast path still needs optional assets and compatible llama-cpp-python wheels. |
| `CambEngine` | `camb_engine.py` | root, engines | `camb`, `all` | `tests/camb_test.py` | Uses `CAMB_API_KEY`. |
| `MiniMaxEngine` | `minimax_engine.py` | root, engines | `minimax`, `all` | `tests/minimax_test.py`, `tests/test_minimax_engine.py`, `tests/test_minimax_integration.py` | Strongest pytest coverage among cloud engines. |
| `CartesiaEngine` | `cartesia_engine.py` | root, engines | `cartesia`, `all` | `tests/cartesia_test.py` | Uses `cartesia==2.0.9`. |
| `TypecastEngine` | `typecast_engine.py` | root, engines | `typecast`, `all` | none focused found | Uses `TYPECAST_API_KEY` and optional `TYPECAST_VOICE_ID`; source arrived from upstream master during release rebase. |
| `FasterQwenEngine` | `faster_qwen_engine.py` | root, engines | `qwen`, `all` | `tests/faster_qwen_test.py`, `tests/faster_qwen_emotions.py` | Extra name is `qwen`, not `faster-qwen`. |
| `OmniVoiceEngine` | `omnivoice_engine.py` | root, engines | `omnivoice`, `all` | `tests/omnivoice_test.py`, `tests/omnivoice_emotions.py` | Voice cloning and inline voice examples need docs. |
| `LuxTTSEngine` | `luxtts_engine.py` | root, engines | `luxtts`, `all` | none focused found | Local checkout/model setup needs verification. |
| `ChatterboxEngine` | `chatterbox_engine.py` | root, engines | `chatterbox`, `all` | none focused found | Extra now uses `chatterbox-tts`; prompt WAV length caveats remain. |
| `SoproTTSEngine` | `sopro_engine.py` | root, engines | `sopro`, `all` | none focused found | Uses Hugging Face cache environment variables when configured. |
| `SopranoEngine` | `soprano_engine.py` | root, engines | `soprano`, `all` | none focused found | Local neural engine without a focused test found. |
| `MossTTSEngine` | `moss_tts_engine.py` | root, engines | `moss`, `moss-tts`, `all` | none focused found | Extra keeps PyPI-resolvable runtime deps; MOSS-TTS-Nano package/assets, CUDA/CUDNN, and model assets still need care. |
| `ModelsLabEngine` | `modelslab_engine.py` | engines only | `modelslab`, `all` | `tests/modelslab_test.py` | README and tests reference it, but root `RealtimeTTS` does not export it. |

## Setup Extras Inventory

Observed `setup.py` extras:

`minimal`, `all`, `system`, `azure`, `elevenlabs`, `openai`, `gtts`, `coqui`,
`edge`, `kokoro`, `camb`, `minimax`, `modelslab`, `cartesia`, `typecast`,
`orpheus`, `omnivoice`, `luxtts`, `zipvoice`, `chatterbox`, `sopro`,
`soprano`, `neutts`, `neutts-gguf`, `pockettts`, `pocket`, `styletts`,
`style`, `parler`, `moss`, `moss-tts`, `piper`, `qwen`, `jp`, `zh`, `ko`.

Packaging mismatches to preserve as follow-ups, not docs assumptions:

- `ModelsLabEngine` is exported from `RealtimeTTS.engines`, but not from the
  root `RealtimeTTS` lazy export table.
- `PiperEngine` still needs an external Piper executable and voice model; setup
  extras cannot install those assets.
- `StyleTTSEngine`, `ZipVoiceEngine`, and some MOSS workflows still need local
  upstream checkouts or model assets even though setup extras now cover their
  declared Python dependency scaffolding.
- `all` is broader now, but remains a best-effort Python dependency set; it
  cannot install OS tools, provider accounts, CUDA wheel choices, or model
  downloads.
- `setup.py` declares Python `>=3.9, <3.15`; `docs/en/usage.md` still says
  Python `<3.13`.
- `setup.py` prepends custom installation guidance to the README for the PyPI
  long description. The README reset should decide whether PyPI uses the compact
  README as-is or a generated long description before editing `setup.py`.

## Runtime Environment Variables

| Variable | Observed in | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI engine, tests, Docker example | OpenAI API credentials. |
| `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` | Azure examples, FastAPI example, tests | Azure Speech credentials. |
| `ELEVENLABS_API_KEY` | ElevenLabs engine, examples, tests | ElevenLabs credentials. |
| `CAMB_API_KEY` | CAMB engine and test | CAMB credentials. |
| `MINIMAX_API_KEY` | MiniMax engine and pytest files | MiniMax credentials. |
| `CARTESIA_API_KEY` | Cartesia engine and test | Cartesia credentials. |
| `MODELSLAB_API_KEY` | ModelsLab engine and test | ModelsLab credentials. |
| `PIPER_PATH` | Piper engine | External Piper executable path. |
| `COQUI_MODEL_PATH`, `TTS_HOME`, `XDG_DATA_HOME` | Coqui engine | Local model discovery/cache paths. |
| `HF_HOME`, `HF_HUB_CACHE` | Sopro engine, ZipVoice Docker | Hugging Face cache paths. |
| `CUDNN_PATH` | MOSS engine | CUDA/CUDNN library discovery. |
| `TTS_FASTAPI_PORT`, `WORKERS` | FastAPI and Docker examples | Server port and worker count. |
| `ZIPVOICE_PROJECT_ROOT`, `VOICE_ALPHA_WAV_PATH`, `VOICE_ALPHA_PROMPT_TEXT`, `VOICE_BETA_WAV_PATH`, `VOICE_BETA_PROMPT_TEXT`, `DEVICE` | ZipVoice Docker server | ZipVoice server configuration. |
| `RVC_ASSET_PATH` and RVC internals | RVC example | RVC demo asset/model locations. |

## Public API Docs Mismatches

These are documentation follow-ups for Stage 3 and Stage 4:

- `docs/en/api.md` omits or under-documents current constructor parameters such
  as `log_characters`, `on_word`, `mpv_audio_device`, `frames_per_buffer`, and
  `playout_chunk_size`.
- Current source includes volume control (`volume` property and `set_volume`),
  inline voice tags (`add_voice`, `set_voice_tag_delimiters`), pause tags
  (`add_pause`), and recursive play behavior that the current docs do not cover.
- `play_async()` defaults `fast_sentence_fragment_allsentences=True`, while
  `play()` defaults it to `False`; docs should call out this difference.
- `force_first_fragment_after_words` defaults to `30` in source, while current
  README/docs material still describes older values in places.
- Sentence delimiter defaults in docs should be re-audited against
  `text_to_stream.py`; the existing docs contain mojibake around ellipsis and
  CJK punctuation.

## Tests And Examples Inventory

| Area | Current state |
| --- | --- |
| Pytest-style tests | `tests/test_minimax_engine.py` and `tests/test_minimax_integration.py` contain real pytest tests. The integration file is gated by `MINIMAX_API_KEY`. |
| Manual scripts named like tests | `tests/test_callbacks.py` and `tests/test_on_audio_chunk_callback.py` execute work at import time and should be documented as manual demos unless refactored later. |
| Engine demos | Most `tests/*_test.py` files are engine demos or smoke scripts, often requiring credentials, local models, audio devices, or optional packages. |
| FastAPI example | `example_fast_api/README.md`, `server.py`, `async_server.py`, and clients are valuable deployment docs inputs. |
| RVC example | `example_rvc/README.md` documents XTTS plus RVC post-processing and should remain clearly framed as an example stack. |
| Docker examples | `docker/zipvoice/README.md` and `docker/openai-tts-docker/Readme.md` should be linked from future Docker and engine docs. |
