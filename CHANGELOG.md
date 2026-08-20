# Changelog

## 0.7.4 (release candidate)

### Added

- Native `QwenEngine` integration through the ABI-4 `qwentts.cpp` binding.
- Installable headless Qwen server via `realtimetts[qwen-server]` and the
  `realtimetts-qwen-server` console command.
- Persistent Qwen voice registration, incremental text streaming, language
  routing, health/readiness reporting, and production request controls.
- `InflectEngine` with pinned PyTorch and ONNX model revisions.
- A shared package version exposed as `RealtimeTTS.__version__`.

### Changed

- Replaced `FasterQwenEngine` with the native `QwenEngine` implementation.
- Made Stanza opt-in and selected the NLTK plus rule-based tokenizer by default.
- Kept the server extra headless while engine playback extras continue to use
  PyAudio/PortAudio.

### Fixed

- Preserved host logging configuration while loading the Inflect runtime.
- Hardened streaming silence trimming and short-utterance startup behavior.
- Added deterministic cancellation, bounded concurrency, and graceful shutdown
  behavior to the Qwen server.

The final release date and verified native-wheel compatibility matrix will be
added when the candidate passes Windows and Linux acceptance.
