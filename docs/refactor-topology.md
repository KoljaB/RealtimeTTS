# Documentation Topology Decision

This is the Stage 1 topology decision for the documentation refactor described
in `roadmap/doc-refacturing.md`. It documents the target docs shape before the
README reset begins.

Inventory date: 2026-05-10.

## Decision

Use canonical English pages at the root of `docs/`, with focused task pages,
reference pages, and per-engine pages:

```text
docs/
  index.md
  quick-start.md
  installation.md
  engine-selection.md
  cloud-credentials.md
  local-gpu-setup.md
  feed-and-playback.md
  llm-streaming.md
  callbacks-and-timing.md
  sentence-splitting.md
  buffering-latency.md
  fallback-engines.md
  output-and-files.md
  voices.md
  voice-cloning.md
  examples.md
  fastapi-server.md
  rvc-post-processing.md
  docker.md
  testing.md
  test-scripts.md
  troubleshooting.md
  licensing.md
  release-notes.md
  migration-notes.md
  contributing.md
  adding-an-engine.md
  translation-strategy.md
  reference/
    text-to-audio-stream.md
    base-engine.md
    stream-player.md
    voice-objects.md
  engines/
    system.md
    azure.md
    elevenlabs.md
    openai.md
    gtts.md
    edge.md
    coqui.md
    piper.md
    styletts.md
    parler.md
    kokoro.md
    orpheus.md
    camb.md
    minimax.md
    cartesia.md
    faster-qwen.md
    omnivoice.md
    modelslab.md
    pockettts.md
    neutts.md
    zipvoice.md
    luxtts.md
    chatterbox.md
    sopro.md
    soprano.md
    moss-tts.md
```

This follows the RealtimeSTT orientation: a compact README, a short working
example near the top, and a documentation map that sends users into focused
pages instead of turning the README into the manual.

## MkDocs Target Nav

Do not implement this nav until the first target pages exist. The next MkDocs
rewrite should move from language-first repetition to topic-first English pages.

```yaml
nav:
  - Home: index.md
  - Start:
      - Quick Start: quick-start.md
      - Installation: installation.md
      - Engine Selection: engine-selection.md
  - Using RealtimeTTS:
      - Feed And Playback: feed-and-playback.md
      - LLM Streaming: llm-streaming.md
      - Output And Files: output-and-files.md
      - Voices: voices.md
      - Voice Cloning: voice-cloning.md
      - Fallback Engines: fallback-engines.md
      - Buffering And Latency: buffering-latency.md
      - Sentence Splitting: sentence-splitting.md
      - Callbacks And Timing: callbacks-and-timing.md
  - Engines:
      - Overview: engine-selection.md
      - System: engines/system.md
      - Azure: engines/azure.md
      - ElevenLabs: engines/elevenlabs.md
      - OpenAI: engines/openai.md
      - gTTS: engines/gtts.md
      - Edge: engines/edge.md
      - Coqui: engines/coqui.md
      - Local And Voice-Cloning Engines:
          - Piper: engines/piper.md
          - StyleTTS: engines/styletts.md
          - Parler: engines/parler.md
          - Kokoro: engines/kokoro.md
          - Orpheus: engines/orpheus.md
          - Faster Qwen: engines/faster-qwen.md
          - OmniVoice: engines/omnivoice.md
          - PocketTTS: engines/pockettts.md
          - NeuTTS: engines/neutts.md
          - ZipVoice: engines/zipvoice.md
          - LuxTTS: engines/luxtts.md
          - Chatterbox: engines/chatterbox.md
          - SoproTTS: engines/sopro.md
          - Soprano: engines/soprano.md
          - MOSS-TTS: engines/moss-tts.md
      - API Engines:
          - CAMB: engines/camb.md
          - MiniMax: engines/minimax.md
          - Cartesia: engines/cartesia.md
          - ModelsLab: engines/modelslab.md
  - Reference:
      - TextToAudioStream: reference/text-to-audio-stream.md
      - BaseEngine: reference/base-engine.md
      - StreamPlayer: reference/stream-player.md
      - Voice Objects: reference/voice-objects.md
  - Examples And Deployment:
      - Examples: examples.md
      - FastAPI Server: fastapi-server.md
      - RVC Post-Processing: rvc-post-processing.md
      - Docker: docker.md
  - Maintenance:
      - Testing: testing.md
      - Test Scripts: test-scripts.md
      - Troubleshooting: troubleshooting.md
      - Licensing: licensing.md
      - Migration Notes: migration-notes.md
      - Release Notes: release-notes.md
      - Contributing: contributing.md
      - Adding An Engine: adding-an-engine.md
      - Translation Strategy: translation-strategy.md
```

## Translation Migration

The English root pages are canonical for the first pass.

Keep existing `docs/<locale>/` pages untouched until the English pages and nav
settle. During the translation stage, either regenerate translated pages from a
known English version or add visible stale-page notices. Do not manually repeat
every locale under every nav topic again.

Before changing `mkdocs.yml`, verify the i18n plugin setup and current encoding
of language names. The existing config output shows mojibake for several labels,
and `docs/ar/` exists even though Arabic is not currently listed in `mkdocs.yml`.

## README Reset Preparation

The README reset should happen after this inventory is reviewed and the first
canonical target pages are created.

The README should keep:

- One short project summary.
- Minimal installation commands and platform notes.
- One first-audio example using a low-friction engine.
- One generator/LLM streaming example.
- One output/file example or callback pointer.
- A compact feature list.
- A compact engine overview table.
- A documentation map into the focused docs pages.
- Short contribution, license, and author sections.

The README should move out:

- Full installation matrix and engine requirements.
- API parameter reference for `TextToAudioStream`, `play`, and `play_async`.
- CUDA, local GPU, and model setup details.
- Historical engine announcements.
- Test script listings.
- Troubleshooting body.
- Engine licensing details.

## PyPI Long Description Decision

Target decision: the compact README should be useful as the PyPI long
description by itself. Do not keep a separate install block prepended in
`setup.py` unless maintainers explicitly want a generated PyPI-specific
description.

Do not edit `setup.py` during Stage 0 or Stage 1. When Stage 2 resets the
README, review the current `setup.py` long-description prepend at the same time
so PyPI does not publish stale engine extras or install guidance.

## First Safe Docs Changes

For this first pass:

- Add this topology decision.
- Add `docs/refactor-source-inventory.md` with the source-of-truth inventory and
  mismatch list.
- Do not rewrite `README.md` yet.
- Do not update `mkdocs.yml` yet.
- Do not edit engine code or packaging code.
- Treat new engine files and export changes currently in the worktree as
  inventory inputs only.

Next safe step after review:

1. Create `docs/index.md`, `docs/quick-start.md`, `docs/installation.md`, and
   `docs/engine-selection.md` as the first canonical English pages.
2. Reset README to link only to pages that exist.
3. Update `mkdocs.yml` once the linked pages are present.
