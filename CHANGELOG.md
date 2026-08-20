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
- Selected the validated native Qwen candidate per platform: Windows uses
  `qwentts-cpp-python` `0.4.0.dev1` (`1cu128`), while Linux uses `0.4.0.dev0`
  (`1cu128`). Unsupported platforms are not declared as candidate targets.

### Fixed

- Preserved host logging configuration while loading the Inflect runtime.
- Hardened streaming silence trimming and short-utterance startup behavior.
- Added deterministic cancellation, bounded concurrency, and graceful shutdown
  behavior to the Qwen server.
- Fixed WebSocket cancellation after the `end` event so active PCM synthesis
  stops promptly and emits the terminal `cancelled` event.

The `0.7.4.dev9` candidate still requires Windows and Linux acceptance and
final versioning of both coordinated packages before stable release.
