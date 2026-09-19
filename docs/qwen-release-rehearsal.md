# Qwen TestPyPI release rehearsal

Qwen GPU/CPU release policy: **TestPyPI first, PyPI only after fresh-install
and real-speech acceptance.** The `0.8.6rc5` framework and `0.3.0rc1` CPU
runtime have passed the four-platform CPU TestPyPI installation/speech matrix
(Linux x86-64, Windows x86-64, Intel Mac, and Apple Silicon), plus Windows GPU
TestPyPI acceptance. Final-version artifacts must pass the gates below before
PyPI publication; candidate results do not replace final-artifact checks.

## Candidate sequence

1. Preserve the existing Linux runtime and all dirty worktree changes. Separate
   the prefix-splice experiment from the production release branch.
2. Build immutable wheel/sdist pairs from clean commits. Use release candidates
   (the historical rehearsal used `realtimetts==0.8.6rc5` and
   `realtimetts-qwen-native-cpu==0.3.0rc1`); the CPU extra must pin the exact
   matching native version. Existing public GPU native wheels can remain pinned.
3. Exercise the exact artifacts in the declared isolated/runtime environments,
   obtain the required fresh signed attestation, and publish to TestPyPI through
   `tools/release_guard.py publish`. TestPyPI does not bypass the parity guard.
4. Download the uploaded artifacts again and verify their index SHA-256 values.
   Test in fresh virtual environments, with no editable install, local source
   shadowing, manual `site-packages` patches, or undeclared native-library paths.
5. If a fix is needed, commit it and build a new candidate (`rc2`, etc.). Uploaded
   filenames cannot be replaced, even if a release is deleted. Keep failed
   candidates and their results as evidence; do not try to overwrite them.
6. Once the candidates pass, build the final-version artifacts once. Publish
   those first to TestPyPI and repeat the installation/runtime acceptance. Then
   publish the **same unchanged final files** to PyPI, with fresh attestation and
   matching public refs. An RC wheel cannot simply be renamed to a final wheel.

## Rehearse the advertised workflow

Keep the original Python 3.11 environment creation and
`cd tests; python faster_qwen_emotions.py` entry point. For the package rehearsal,
replace `pip install -e .[qwen]` with installation of the exact uploaded
candidate; an editable installation would mask missing packaged files.
The checkout supplies only the historical script and original reference WAVs.
Verify that `RealtimeTTS.__file__` resolves inside the fresh environment's
`site-packages`, not inside the checkout.

Fetch only the explicitly versioned RealtimeTTS/native candidate wheels from
TestPyPI (`pip download --no-deps --only-binary=:all:`). Install those downloaded
wheels with their selected extras while resolving ordinary dependencies from
regular PyPI. This avoids treating all TestPyPI projects as trusted dependency
sources. Record versions, wheel hashes, Python version, platform, native ABI,
model hashes, launch settings, and the saved synthesis results.

The acceptance matrix must cover:

- Windows/Linux NVIDIA: `qwen-server`, server HTTP and WebSocket synthesis, and
  the original emotional demo with local playback dependencies.
- Windows/Linux CPU: `qwen-cpu-server`, the same server protocols, and the CPU
  emotional demo with the deployed onset profile/recovery enabled.
- macOS CPU: platform-specific wheel installation and actual synthesis on hosted
  macOS runners. An import-only test does not establish speech support, and
  headless CI does not establish audible local-device playback.
- Package-only use without a checkout: installed demo entry point, pinned
  reference download/checksums, and nonempty saved speech.
- Production Linux parity: em-dash early speech, `flush`, `segment_start`,
  `skip_segment`, language detection, voice/model configuration, CPU recovery,
  and the preserved native worker/affinity changes.

The two public install commands must resolve directly from PyPI after release;
users must not need TestPyPI, a private wheelhouse, or a source compiler on the
supported wheel platforms.

References: [Using TestPyPI](https://packaging.python.org/en/latest/guides/using-testpypi/)
and [PyPI filename reuse policy](https://pypi.org/help/#file-name-reuse).
