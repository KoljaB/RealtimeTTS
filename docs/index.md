# RealtimeTTS Documentation

RealtimeTTS streams text into speech with a small Python API. It can speak
plain strings, generators, and LLM token streams, then play audio locally, write
audio to a WAV file, or hand chunks to another process.

English is the canonical source for this documentation refactor. Existing
translated pages remain in `docs/<locale>/` until the translation workflow is
updated.

## Start Here

- [Quick start](quick-start.md): install one engine and hear audio.
- [Installation](installation.md): extras, platform notes, external tools, and
  known packaging mismatches.
- [Engine selection](engine-selection.md): choose a local, cloud, voice-cloning,
  or experimental engine.

## Core Workflows

- [Feed and playback](feed-and-playback.md): `feed()`, `play()`, `play_async()`,
  pause, resume, stop, and fallback basics.
- [LLM streaming](llm-streaming.md): connect streamed text chunks to
  RealtimeTTS without waiting for a full response.
- [Output and files](output-and-files.md): output devices, mpv playback, muted
  mode, WAV output, and audio chunk callbacks.

## Examples

- `tests/simple_test.py` is the smallest local script.
- `tests/simple_llm_test.py` shows an older LLM streaming pattern and should be
  modernized before being promoted as canonical.
- `example_fast_api/` contains HTTP and WebSocket server examples.
- `example_rvc/` contains an XTTS plus RVC post-processing example.
- `docker/zipvoice/` and `docker/openai-tts-docker/` contain Docker examples.

The examples under `tests/` are a mix of demos, manual smoke tests, and a small
number of pytest tests. Treat `tests/test_minimax_engine.py` and
`tests/test_minimax_integration.py` as the current maintained pytest-style
pattern.

## Project And License

RealtimeTTS source code is MIT licensed. Engine providers, model files, voices,
datasets, generated audio, and third-party services can have separate terms.
Read `LICENSING_ADDENDUM.md` and the provider/model licenses before commercial
use.
