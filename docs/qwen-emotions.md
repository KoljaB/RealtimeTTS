# Emotional Qwen showcase

`tests/faster_qwen_emotions.py` is the supported entry point advertised in the
emotional Qwen video. Its old filename is intentional. It now uses the maintained
native Qwen backend, with the original eleven reference recordings and spoken
texts, 0.6B Base Q8, and speaker-only cloning. Some original showcase texts contain
strong language. Emotion comes from the recording, not an instruction prompt.

## Release and platform support

These commands target RealtimeTTS 0.8.6 and CPU native runtime 0.3.0. Use
Python 3.11 or 3.12 for the Qwen workflows. Core-package Python support does
not imply that every optional audio/native dependency supports newer Python.

GPU mode supports NVIDIA GPUs on Windows/Linux x86-64. CPU wheels cover
Windows x86-64, Linux x86-64 with glibc 2.35+, Intel macOS 13+, and Apple
Silicon macOS 11+ (use Python matching the machine architecture). x86-64
requires AVX2, FMA, F16C, and BMI2; older CPUs, Windows ARM64, Linux ARM CPU,
and Metal acceleration are outside this release. CPU speed depends on the
machine: installation support is not a guarantee of faster-than-realtime audio.

Publication is gated on fresh non-editable TestPyPI installations and real
synthesis, followed by publication of the same final artifacts to PyPI. See
the [TestPyPI acceptance plan](qwen-release-rehearsal.md).

## GPU: the existing video command

Create and activate a Python 3.11 virtual environment. On Windows cmd:

```bat
py -3.11 -m venv test_env
call test_env\Scripts\activate.bat
python -m pip install --upgrade pip
git clone https://github.com/KoljaB/RealtimeTTS.git
cd RealtimeTTS
python -m pip install -e ".[qwen]"
cd tests
python faster_qwen_emotions.py
```

On Linux/macOS the environment commands are `python3 -m venv test_env` and
`source test_env/bin/activate`. GPU mode requires a supported NVIDIA GPU and
driver on Windows/Linux. Native wheels install the CUDA runtime dependencies;
manual Torch, torchvision, torchaudio and CUDA Toolkit installation is unnecessary.
For local speaker playback Linux needs PortAudio (for example `portaudio19-dev` on
Ubuntu); macOS needs PortAudio (for example `brew install portaudio`). Windows
Python 3.11 has a prebuilt PyAudio wheel.

The maintained CPU equivalent uses `python -m pip install -e ".[qwen-cpu]"` and
`python faster_qwen_emotions.py --device cpu`.
The model is not silently changed to 1.7B or another backend. The local CPU demo
also enables the deployed CPU onset-silence profile and recovery. The CPU server
command below enables those explicitly. Exact production reproduction additionally
requires the same model hashes, voices, seed, thread/affinity settings and launch
configuration; an installation alone does not reproduce machine-specific settings.

## Server installs

In separate fresh environments, the release installation targets are:

```bash
python -m pip install "realtimetts[qwen-server]"
realtimetts-qwen-server --host 127.0.0.1 --port 8080 --clone-mode speaker_only
```

```bash
python -m pip install "realtimetts[qwen-cpu-server]"
realtimetts-qwen-server --device cpu --host 127.0.0.1 --port 8080 --clone-mode speaker_only --no-clamp-fp16 --onset-silence-profile qwen3_tts_12hz_0_6b_base_q8_v1 --onset-silence-recovery
```

The server extras do not install local sound-device dependencies. Open the
server's `/studio` page for browser playback. To exercise the original emotional
references against either server without needing an audio device:

```bash
python faster_qwen_emotions.py --server http://127.0.0.1:8080 --no-play
```

With only a package installation (no Git clone), the equivalent command is:

```bash
python -m RealtimeTTS.qwen_emotions --server http://127.0.0.1:8080 --no-play
```

The installed module downloads the original public reference WAVs from a pinned
repository revision into the user cache and verifies SHA-256 hashes. A checkout
uses the tracked WAVs beside the historical script. No `.pt` conversion or Torch
cache is required. Voice preparation is cached by the native engine.

For local playback from a server client install `realtimetts[playback]`, then omit
`--no-play`. Pass `--api-key-file PATH` or set `REALTIMETTS_API_KEY` if the server
requires authentication; do not put keys in URLs. The server must load the
original 0.6B Base model. Demo registrations have content-derived names prefixed
with `emotions-demo-`; they do not replace the server's normal voices.

## Quick checks and output

```bash
python faster_qwen_emotions.py --list
python faster_qwen_emotions.py --emotions neutral anger --text "This is a short test." --no-play
```

Every successful run saves one nonempty 24 kHz mono WAV per emotion and a
`results.json` in `qwen_emotions_output` (override with `--output-dir`). Local
playback reports time to playback start; the HTTP client reports time to first
received PCM, not audible speaker latency. First-time model download/preparation
is distinct from warm synthesis latency. `--local-files-only` disables downloads;
missing required files fail explicitly. `--model` and `--codec` can select local
copies of the same model pair. The native `.spk`/`.rvq` cache does not reuse old
Torch `.pt` caches.
