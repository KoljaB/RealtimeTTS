# Qwen Engine

`QwenEngine` runs Qwen3-TTS in-process through `qwentts.cpp`. It keeps one
native model context resident, streams 24 kHz mono PCM directly into the
RealtimeTTS queue, and supports cached x-vector and full ICL voice cloning.
The optional HTTP server wraps that same in-process engine without adding a
second model process.

Leading-silence trimming is enabled by default. It finds the first audible
5 ms window, keeps 15 ms of pre-roll before it, and applies a 20 ms fade at the
new boundary. The engine accumulates at least 160 ms of real audio before
publishing the first chunk, preventing immediate-play underruns after making
the first native frame shorter. Later native chunks are published immediately.
Short utterances are flushed in full even when they never reach 160 ms. Use
`QwenEngine(trim_silence=False)` to preserve the native PCM, set
`startup_buffer_ms=0` to disable startup accumulation, or tune
`silence_threshold`, `trim_pre_roll_ms`, and `trim_fade_in_ms` for unusual
material.

## Release status and installation

Qwen support is a coordinated release of two packages: `RealtimeTTS` and
`qwentts-cpp-python`. This engine requires the native package's C ABI 4 and a
`qwentts-cpp-python` version in the range declared by the matching RealtimeTTS
release. Do not combine this engine with the older 0.3.x native package.

For the current `0.7.4.dev8` TestPyPI candidate, the validated CUDA 12.8
(`1cu128`) wheel differs by platform: Windows x86-64 uses
`qwentts-cpp-python==0.4.0.dev1`, while Linux x86-64 uses
`qwentts-cpp-python==0.4.0.dev0`. The Qwen extras declare native requirements
only for those Windows and Linux targets; other operating systems are not a
supported candidate target and must not be treated as a working Qwen install.

When the matching release is on PyPI, install the normal CUDA wheel with:

```bash
python -m pip install --only-binary=qwentts-cpp-python "realtimetts[qwen]"
python -m qwentts_cpp doctor
```

`--only-binary=qwentts-cpp-python` makes a missing platform wheel fail clearly;
there is no qwentts source-distribution fallback. If no wheel matches your
platform, Python, or architecture, build a wheel as described in [Build a
local native wheel](#build-a-local-native-wheel), then install it from a
wheelhouse. A repaired wheel does not need a compiler or CUDA Toolkit at
runtime; it does need a compatible NVIDIA driver for CUDA builds.

For pre-release validation from TestPyPI, resolve dependencies from normal
PyPI while taking both coordinated candidate wheels from TestPyPI. Install the
native candidate with `--no-deps` first so pip cannot silently select the old
0.3.x ABI from public PyPI. The final command keeps PyPI as the dependency
index and uses the TestPyPI project page only as a find-links source for the
exact RealtimeTTS candidate:

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
  "realtimetts[qwen-server]==0.7.4.dev8"
python -m qwentts_cpp doctor
python -m pip check
```

The coordinated validation pair is RealtimeTTS `0.7.4.dev8` with the
platform-specific native candidate listed above. Treat the pair as supported
only when that platform's wheel is present on TestPyPI and passes
`python -m qwentts_cpp doctor`; do not substitute the older `0.4.0.dev0`
Windows `1cu125` artifact, or use the Linux `0.4.0.dev0` pin on Windows.

For a local RealtimeTTS wheel and a locally built native wheel, keep both files
in a wheelhouse (or point the RealtimeTTS requirement at its absolute wheel
path) so pip cannot silently select a different public version:

```bash
python -m pip install --find-links /absolute/path/to/native-wheelhouse \
  "qwentts-cpp-python[cuda12]==0.4.0.dev1"
python -m pip install --find-links /absolute/path/to/native-wheelhouse \
  "realtimetts[qwen] @ file:///absolute/path/to/realtimetts-wheel.whl"
python -m qwentts_cpp doctor
```

For the OpenAI-compatible server, install the server extra instead:

```bash
python -m pip install "realtimetts[qwen-server]"
```

Both extras retain RealtimeTTS's default `nltk+rule-based` sentence tokenizer
and do not install Stanza or PyTorch. The server extra does not install
PyAudio because it returns PCM over HTTP and never opens a local audio device.
The HTTP speech endpoint intentionally
synthesizes each submitted `input` as one request, matching the native
qwentts.cpp server contract; it does not split that text into sentence seams.

This extra uses PyAudio/PortAudio for supported local playback. On Windows use
Python 3.10–3.13, for which PyAudio publishes prebuilt wheels. On Linux install
`portaudio19-dev` before this extra; on macOS install `portaudio` with Homebrew.
Python 3.13 also installs the prebuilt `audioop-lts` compatibility module needed
by pydub.

## Native wheel compatibility

The native Qwen binding is `ctypes`-based rather than a CPython extension, so
its `py3-none` wheel can load on Python 3.10 through 3.14. Supported RealtimeTTS
local playback on Windows is narrower—Python 3.10–3.13—because PyAudio 0.2.14
does not publish a Windows Python 3.14 wheel.
RealtimeTTS supports Python 3.9 for some other engines; that does not make the
Qwen extra installable on Python 3.9.

| Platform | Status/requirement |
| --- | --- |
| Windows 10/11 x86-64 | Current `0.7.4.dev8` candidate: `0.4.0.dev1`, `1cu128`, `py3-none-win_amd64`; AVX2/FMA/F16C/BMI2 CPU; NVIDIA GPU with compute capability 7.5 or newer; CUDA-12-compatible driver. This is the primary Windows release target. |
| Linux x86-64 | Current `0.7.4.dev8` candidate: `0.4.0.dev0`, `1cu128`, `py3-none-manylinux_2_35_x86_64`; glibc 2.35 or newer (Ubuntu 22.04/24.04); AVX2/FMA/F16C/BMI2 CPU; NVIDIA GPU with compute capability 7.5 or newer. This is the primary Linux release target. |
| Linux AArch64 | `py3-none-manylinux_2_35_aarch64` when a matching artifact is published; secondary/target-dependent, not guaranteed by every RealtimeTTS release. |
| Linux CPU | A locally built `manylinux` wheel is possible, but CPU realtime performance is not a supported release guarantee. |
| macOS / Apple Silicon | No supported prebuilt wheel. `qwentts.cpp` itself has a Metal backend, but the Python wheel helper has no `metal` backend and its macOS library-copy list does not include `libggml-metal.dylib`; the route below is an unverified CPU-only experiment. |
| CUDA | The default published CUDA wheel is built with CUDA 12.8. The `[cuda12]` extra supplies NVIDIA's `nvidia-cuda-runtime-cu12` and `nvidia-cublas-cu12` packages (`>=12.8,<13`); a compatible NVIDIA driver is still required. |

AMD/Vulkan, Alpine/musl, Windows ARM64, and guaranteed realtime CPU synthesis
are not part of the supported Qwen release. CUDA libraries and model weights
make this a large download. The current default Q8 talker and codec total about
1.3 GB; exact sizes can change in the upstream model repository.

The CUDA wheel contains qwen/GGML native libraries but deliberately does not
bundle model weights, the NVIDIA driver, CUDA runtime, or cuBLAS. The runtime
extra supplies the latter two Python packages. You do not need Visual Studio,
CMake, Ninja, NVCC, or a full CUDA Toolkit to install a supported wheel; those
tools are needed only when building one yourself. CUDA release binaries use
portable CPU code (`GGML_NATIVE=OFF`) and include GPU code for compute
capability 7.5 and newer (native cubins and/or PTX depend on the target wheel).

The upstream project can also publish backend-specific local-version wheels such
as `+cpu`, `+cu124`, `+cu128`, and `+cu130` through its Hugging Face wheel
index. Those variants are useful for a private wheelhouse, but are not a
substitute for the single default backend flavor published to PyPI.

## Build a local native wheel

Use the upstream [`qwentts-cpp-python`](https://github.com/andimarafioti/qwentts-cpp-python)
`scripts/build_native.py` helper. Run these commands from that repository, and
keep build output outside its checkout. The wrapper checks the native header
before CMake starts; use the ABI-4 qwentts.cpp revision tested by the matching
release (currently `7b6ed4f6db964c14fd3ac36c1ca13f1ce6150f4e`):

```bash
git clone --recursive https://github.com/ServeurpersoCom/qwentts.cpp /path/to/qwentts.cpp
git -C /path/to/qwentts.cpp checkout 7b6ed4f6db964c14fd3ac36c1ca13f1ce6150f4e
git -C /path/to/qwentts.cpp submodule update --init --recursive
```

### Windows x86-64 CUDA

Install Python 3.10–3.14, Visual Studio Build Tools with the MSVC workload,
CUDA Toolkit 12.8, and initialize a Visual Studio developer environment. Then
install the Python build tools and run the same native build/repair sequence as
the upstream Windows workflow:

```powershell
python -m pip install --upgrade pip build cmake delvewheel ninja wheel
$env:CUDACXX = "$env:CUDA_PATH\bin\nvcc.exe"
python scripts/build_native.py `
  --source D:\src\qwentts.cpp `
  --backend cuda `
  --clean `
  --cmake-arg=-G `
  --cmake-arg=Ninja `
  --cmake-arg="-DCMAKE_CUDA_ARCHITECTURES=75-real;75-virtual"
$env:QWENTTS_CPP_WHEEL_BUILD_TAG = "1cu128"
python -m build --wheel
python -m delvewheel repair `
  --analyze-existing `
  --ignore-existing `
  --add-path "$env:CUDA_PATH\bin;$env:CUDA_PATH\bin\x64;src\qwentts_cpp\lib" `
  --exclude "cudart64_12.dll;cublas64_12.dll;cublasLt64_12.dll;nvcuda.dll" `
  --wheel-dir D:\wheelhouse `
  dist\*.whl
```

The resulting `py3-none-win_amd64` wheel still requires a CUDA-12-compatible
driver at runtime. The CUDA runtime and cuBLAS remain external because the
`cuda12` extra provides them. Use `--source` for an existing qwentts.cpp
checkout; do not build against an arbitrary ABI revision.

### Linux x86-64 CUDA

Build on Ubuntu 22.04 (glibc 2.35) or a compatible manylinux build environment
with Python 3.10–3.14, CMake, Ninja, `patchelf`, and the CUDA 12.8 development
toolkit (`nvcc`, CUDA runtime development files, and cuBLAS development files):

```bash
python -m pip install --upgrade pip auditwheel build wheel
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
python scripts/build_native.py \
  --source /path/to/qwentts.cpp \
  --build-dir /artifacts/qwentts-build \
  --backend cuda \
  --clean \
  --cmake-arg=-G \
  --cmake-arg=Ninja \
  --cmake-arg='-DCMAKE_CUDA_ARCHITECTURES=75-virtual;86-real;90-real;120-real;120-virtual'
QWENTTS_CPP_WHEEL_BUILD_TAG=1cu128 \
  python -m build --wheel --outdir /artifacts/raw
python -m auditwheel repair \
  --plat manylinux_2_35_x86_64 \
  --exclude libcudart.so.12 \
  --exclude libcublas.so.12 \
  --exclude libcublasLt.so.12 \
  --exclude libcuda.so.1 \
  --wheel-dir /artifacts/repaired \
  /artifacts/raw/*.whl
```

The repaired artifact should have a `py3-none-manylinux_2_35_x86_64` tag. Keep
it below PyPI's per-file limit, run `python -m twine check --strict`, and test
it from a fresh environment. Do not upload a development wheel under a final
version name; change the package version and rebuild for a release.

For a Linux CPU experiment, use the same helper with `--backend cpu`, set the
local build tag to `1cpu`, and repair with `auditwheel` for the target
`manylinux` tag. This is useful when CUDA is unavailable, but it is not a
RealtimeTTS realtime-performance promise:

```bash
python scripts/build_native.py \
  --source /path/to/qwentts.cpp \
  --backend cpu \
  --clean \
  --cmake-arg=-G \
  --cmake-arg=Ninja
QWENTTS_CPP_WHEEL_BUILD_TAG=1cpu python -m build --wheel --outdir /artifacts/raw
python -m auditwheel repair \
  --plat manylinux_2_35_x86_64 \
  --wheel-dir /artifacts/repaired \
  /artifacts/raw/*.whl
```

### macOS / Apple Silicon (unverified)

There is no published or release-tested macOS Python wheel. The following is a
best-effort CPU-only build for an Apple Silicon Python interpreter. The
explicit `GGML_METAL=OFF` matters because qwentts.cpp enables Metal by default
on Apple platforms, while the current Python helper does not package its Metal
backend library:

```bash
python -m pip install --upgrade pip build cmake ninja wheel
python scripts/build_native.py \
  --source /path/to/qwentts.cpp \
  --backend cpu \
  --clean \
  --cmake-arg=-G \
  --cmake-arg=Ninja \
  --cmake-arg=-DGGML_METAL=OFF
QWENTTS_CPP_WHEEL_BUILD_TAG=1cpu \
  python -m build --wheel --outdir /absolute/path/to/mac-wheelhouse
```

Install the resulting host-specific `macosx_*_arm64` wheel only into an arm64
Python environment and test it in a fresh environment. `python -m qwentts_cpp
doctor` is expected to report macOS as unsupported by the current diagnostics;
that is why this path is documented as unverified. A working Metal wheel would
require upstream packaging changes to add a Metal backend and bundle/load
`libggml-metal.dylib`; do not assume that a successful CMake build provides a
working RealtimeTTS wheel.

After any local build, install only from the repaired wheelhouse and verify the
native ABI before trying RealtimeTTS. Use the CUDA extra only for a CUDA wheel;
CPU and macOS wheels do not need it:

```bash
python -m venv /path/to/fresh-venv
# Windows x86-64 CUDA candidate:
/path/to/fresh-venv/bin/python -m pip install \
  --find-links /path/to/wheelhouse \
  "qwentts-cpp-python[cuda12]==0.4.0.dev1"
# Linux x86-64 CUDA candidate (use this line instead on Linux):
/path/to/fresh-venv/bin/python -m pip install \
  --find-links /path/to/wheelhouse \
  "qwentts-cpp-python[cuda12]==0.4.0.dev0"
# Linux CPU or macOS CPU wheel (use this line instead of the CUDA line above):
/path/to/fresh-venv/bin/python -m pip install \
  --find-links /path/to/wheelhouse \
  "qwentts-cpp-python==0.4.0.dev1"
/path/to/fresh-venv/bin/python -c \
  "from qwentts_cpp import QwenLibrary, QT_ABI_VERSION; print(QwenLibrary().version(), QT_ABI_VERSION)"
```

On Windows, use `fresh-venv\Scripts\python.exe`. Then install the matching
RealtimeTTS wheel from the same wheelhouse and exercise model loading, streaming,
cancellation, x-vector cloning, ICL cloning, and repeated requests on the
target machine. A shared-library load check alone is not release validation.

## Quick start

```python
from RealtimeTTS import QwenEngine, QwenVoice, TextToAudioStream


voice = QwenVoice(
    name="narrator",
    ref_audio="reference.wav",
    ref_text="The exact words spoken in reference.wav.",
    language="english",
)

engine = QwenEngine(voice=voice)
try:
    stream = TextToAudioStream(engine)
    stream.feed("Native Qwen speech starts streaming as soon as frames arrive.")
    stream.play()
finally:
    engine.shutdown()
```

Defaults are Qwen3-TTS 12 Hz 0.6B Base, Q8_0, one persistent native context,
and 24 kHz mono signed 16-bit output.

## What streams, and when

Qwen streams audio frames, but it does not synthesize from an endlessly growing
token stream. RealtimeTTS first submits a complete text segment (normally a
sentence or configured fragment) to `QwenEngine`. The native qwentts.cpp loop
then calls the Python binding as each autoregressive frame is generated; the
stateful codec emits 1,920 samples per frame, about 80 ms at 24 kHz. Playback
can therefore start before the complete segment's waveform exists, while text
that has not yet formed a complete segment cannot affect that in-flight native
request. This is the same frame-by-frame callback path described in the
upstream [qwentts.cpp architecture notes](https://github.com/ServeurpersoCom/qwentts.cpp/blob/master/docs/ARCHITECTURE.md).

## HTTP server

The server uses the same model, codec, voice latents, sampling controls, and
24 kHz PCM stream as `QwenEngine`. A direct-GGUF launch looks like this:

```bash
realtimetts-qwen-server \
  --model /models/qwen-talker-0.6b-base-Q8_0.gguf \
  --codec /models/qwen-tokenizer-12hz-Q8_0.gguf \
  --alias qwen3-tts-native-q8 \
  --voice-dir /var/lib/realtimetts/qwen-voices \
  --host 127.0.0.1 \
  --port 18084 \
  --lang English
```

Use `python -m RealtimeTTS.qwen_server` when the console script is not on
`PATH`. Omit `--model` and `--codec` to use `--model-id` plus `--quant` and the
Hugging Face cache. The process always runs one Uvicorn worker because a second
worker would load a second model and duplicate VRAM.

The default `--host 127.0.0.1` and explicit localhost CORS origins are
intentional. Wildcard CORS (`*`) is rejected; set one or more deliberate
`--cors-origin` values for browser clients. The server has built-in bearer-key
authentication: provide `--api-key`, preferably `--api-key-file`, or
`REALTIMETTS_API_KEY`. The key is accepted as `Authorization: Bearer ...` (or
the WebSocket `api_key` query parameter when a client cannot set headers).

Non-loopback binds are refused unless both `--allow-lan` and an API key are
present. For a deliberate, trusted-LAN-only bind:

```bash
install -m 600 /dev/null /etc/realtimetts/qwen-api-key
realtimetts-qwen-server \
  --host 0.0.0.0 \
  --allow-lan \
  --api-key-file /etc/realtimetts/qwen-api-key \
  --cors-origin https://tts.example \
  --model /models/qwen-talker-0.6b-base-Q8_0.gguf \
  --codec /models/qwen-tokenizer-12hz-Q8_0.gguf
```

This direct listener is still plain HTTP. For production, keep Qwen bound to
127.0.0.1 and put it behind a trusted reverse proxy that terminates TLS and
enforces your API-key or equivalent authentication policy. Forward
`Authorization`, `X-Request-ID`, `X-Session-ID`, and WebSocket upgrade headers;
do not expose an unauthenticated LAN port or treat CORS as an access-control
boundary. A minimal systemd shape is:

```ini
[Service]
ExecStart=/opt/realtimetts/.venv/bin/realtimetts-qwen-server --host 127.0.0.1 --port 18084 --api-key-file /etc/realtimetts/qwen-api-key --voice-dir /var/lib/realtimetts/qwen-voices
Restart=on-failure
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/var/lib/realtimetts
```

Terminate TLS and apply the external auth policy at the reverse proxy before
forwarding to `http://127.0.0.1:18084`. The built-in key remains useful as a
second boundary even when the proxy is trusted.

The API returns 24 kHz mono signed-16 PCM (or a WAV wrapper), and concurrency is
deliberately one because a model context serializes synthesis. No model weights,
voice latents, or reference audio are bundled in the Python package. Download
or supply those assets separately, and use reference audio only when you have
the speaker's permission and the rights required for your application.

### Licensing and third-party boundaries

RealtimeTTS is MIT-licensed. The `qwentts.cpp` native library and
`qwentts-cpp-python` binding are MIT-licensed. The Qwen 0.6B Base model and its
tokenizer are Apache-2.0. Inflect Micro-v2 and its ONNX artifact are
Apache-2.0. These model and native dependencies are installed or downloaded
separately; none are bundled in a RealtimeTTS sdist or wheel. Check the
upstream license files and model terms for any additional assets you add, and
obtain the necessary rights for every voice, reference recording, and generated
voice latent before deployment.

The server uses the same safe streaming-start defaults as the local engine:
15 ms onset pre-roll, a 20 ms boundary fade, and at least 160 ms in its first
PCM response chunk. Configure these with `--trim-pre-roll-ms`,
`--trim-fade-in-ms`, and `--startup-buffer-ms`; use `--no-trim-silence` or
`--startup-buffer-ms 0` only when comparing raw native output or latency.

### Automatic language and reference routing

For multilingual ICL voices, `Auto` can classify each complete TTS fragment
locally with fastText and select both Qwen's explicit language and a matching
registered reference voice:

```bash
realtimetts-qwen-server \
  ... \
  --lang Auto \
  --language-id-model /models/fasttext/lid.176.ftz \
  --voice-language-route mira:de=mira_de \
  --voice-language-route mira:en=mira
```

`lid.176.ftz` must already exist locally; the server never downloads a model
while handling a request. Explicit request languages such as `German` or
`English` bypass detection but still select their configured reference. The
detector accepts only sufficiently long, sufficiently scored, well-separated results;
ambiguous short fragments use the detector's configured fallback. Without
`--language-id-model`, `Auto` retains the native Qwen behavior.

Between a top score of `0.50` and `0.67`, the required Top-1/Top-2 margin is
interpolated continuously from `0.35` down to `0.20`. Scores below `0.50` are
rejected; scores at or above `0.67` retain the `0.20` margin floor. This avoids
hard confidence buckets while keeping weak, broadly distributed predictions
out of voice routing.

The API intentionally matches qwentts.cpp's server routes:

- `GET /health`
- `GET /ready`
- `GET /v1/capabilities`
- `GET /v1/models`
- `GET`, `POST`, and `DELETE /v1/audio/voices`
- `POST /v1/audio/speech` with `pcm` or `wav` output
- `WS /v1/audio/speech-stream` for incremental text and PCM

`/v1/capabilities` is the source of truth for the server protocol, audio
metadata, limits, and endpoint list. `X-Request-ID` is the canonical request
identifier on every HTTP success or error response. Audio responses also include
`X-Audio-Encoding`, `X-Audio-Sample-Rate`,
`X-Audio-Channels`, and `X-Audio-Bits-Per-Sample`. The baseline is 24,000 Hz,
mono, signed little-endian PCM16 (`audio/pcm`) or the equivalent WAV wrapper;
the single model context permits one active synthesis request at a time.

The WebSocket stream starts with a JSON `config` event containing a registered
`voice` and optional `language`, `instructions`, sampling fields, and
`response_format: "pcm"`. Send JSON `text` events as text arrives, then one
`end` event; send `cancel` to stop the current request. The server emits
`fragment_ready` and `first_pcm_ready` JSON events before binary PCM chunks,
then a terminal `done`, `cancelled`, or `error` JSON event. Every event carries
the `session_id` and request-level `request_id` when applicable; readiness and
first-audio events also carry the exact audio metadata and timing fields.

The protocol is resumeless. A dropped socket or a client reconnect starts a
fresh session: create a new connection, send a new `config`, resend the text
that must be synthesized, and use the new session/request IDs. Do not append to
an old session or assume the server will replay already-emitted PCM.

Voice registration accepts either `wav_b64` or both `spk_b64` and `rvq_b64`,
plus optional `ref_text` for ICL cloning. Registrations are atomically stored
under `--voice-dir` and survive process restarts. Loaded native voice refs are
also cached in memory, so switching among registered voices does not repeatedly
parse the latent files.

Set `--startup-warmup-voice NAME` to prepare one existing persistent voice and
run a short hidden native synthesis before the ASGI application becomes ready.
The voice must already exist under `--voice-dir`; a missing voice fails startup
instead of silently leaving the first user request cold. Hidden warmup audio is
discarded and does not affect HTTP response queues or request metrics. Override
the default warmup sentence and 32-token limit with `--startup-warmup-text` and
`--startup-warmup-tokens`. Add the option on a subsequent launch after the voice
has been registered; omit it during the initial registration launch or to retain
lazy startup.

For PCM streaming, the server buffers output until a block reaches an average
absolute signed-16 amplitude of 80. Leading silence is preserved once speech
starts. If the whole request remains silent, the server returns HTTP 502 with
an OpenAI error envelope of type `output_error`; it never exposes a misleading
empty `200` stream. Backend failures use `server_error` and do not by themselves
mark the process unhealthy.

`/health` reports `active_requests`, `requests_total`,
`synthesis_failures_total`, `unusable_outputs_total`,
`last_progress_age_ms`, and `stalled`. It returns HTTP 503 with
`status=degraded` only while an active request has made no PCM progress for 30
seconds. This is designed for a five-second watchdog that restarts after three
consecutive failed health checks; a single silent output leaves health at 200.
Change the stall interval with `--stall-timeout` when needed.

CUDA Graphs, Flash Attention, and other native compute paths are not disabled
by the server. They follow the native wheel and environment. In particular,
setting `GGML_CUDA_DISABLE_GRAPHS=1` still disables graphs for parity tests or
VRAM troubleshooting; leave it unset for the normal optimized path unless the
deployment has a specific reason to disable them.

## X-vector and ICL cloning

- Leave `ref_text=None` to use only the speaker embedding (x-vector). This is
  useful when the target language differs from the reference.
- Supply the exact reference transcript to use full in-context-learning (ICL)
  cloning. The engine sends both `.spk` and `.rvq` conditioning and usually
  preserves the reference voice more closely.

The transcript must describe the reference recording, not the new synthesis
text. Whitespace-only transcripts are treated as absent.

Existing native files are also accepted:

```python
voice = QwenVoice(
    "cached",
    spk_path="narrator.spk",
    rvq_path="narrator.rvq",
    ref_text="The exact reference transcript.",
)
```

Both files are required. With no `ref_text`, RVQ codes are loaded but omitted
from synthesis so the mode remains x-vector. Pre-encoded files are specific to
the talker architecture that produced them; regenerate them from the reference
audio if the native backend reports a speaker-dimension mismatch.

## Voice cache and offline use

Reference WAVs are decoded, converted to mono, resampled to 24 kHz, and encoded
once. Generated `.spk`/`.rvq` data is committed atomically; the metadata file is
written last. A cache entry is accepted only when its content hashes match.

The key includes:

- reference-audio SHA-256;
- normalized reference text;
- model and quantization;
- native ABI and backend version;
- Python binding version and cache format.

Defaults are `%LOCALAPPDATA%\RealtimeTTS\qwen\voices` on Windows and
`${XDG_CACHE_HOME:-~/.cache}/realtimetts/qwen/voices` on Linux. Override it
with `voice_cache_dir`. Legacy PyTorch `.pt` embeddings are intentionally not
compatible; recreate them from the source WAV.

For model-cache prefetch and offline validation, use the native package CLI:

```bash
python -m qwentts_cpp prefetch --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --quant Q8_0
python -m qwentts_cpp doctor
```

Then construct the engine with `local_files_only=True`. These commands require
the matching 0.4.x native package and may be unavailable in older local wheels.

## Sampling and model options

`QwenEngine` exposes `seed`, `max_new_tokens`, `do_sample`, `temperature`,
`top_k`, `top_p`, `repetition_penalty`, and the four `subtalker_*` overrides.
They can be supplied to the constructor or changed between requests with
`set_voice_parameters()`.

Native/model options include `model_id`, `quant`, `model_cache_dir`,
`local_files_only`, `library_path`, `use_fa`, `clamp_fp16`, and
`codec_chunk_sec`. `max_batch=1` is deliberate: a model context serializes
syntheses for deterministic cache and cancellation behavior.

`clamp_fp16=True` is the default. The native backend documents this as an
FP16-overflow guard for pre-Ampere NVIDIA GPUs, including Turing cards such as
the RTX 20 series. Disable it only after validating output quality on the
target GPU; on an RTX 2080 SUPER it added about 1.8% synthesis time in local
testing. The installation smoke test uses seed 42 so repeated runs are directly
comparable; pass `--seed -1` when intentionally testing random sampling.

`codec_chunk_sec` defaults to the native 24-second exact buffered-decode
window. Native streaming uses its persistent frame decoder and ignores this
buffered-path memory/quality setting.

For GGUF files outside the Hugging Face cache layout, provide both direct paths:

```python
engine = QwenEngine(
    talker_path="D:/models/qwen-talker-0.6b-base-Q8_0.gguf",
    codec_path="D:/models/qwen-tokenizer-12hz-Q8_0.gguf",
    voice=voice,
)
```

The hashes of explicit GGUF files become part of generated voice-cache keys.

## Warmup, threading, cancellation, and shutdown

With `warmup=True`, setting a voice consumes a short native synthesis without
putting its audio in the playback queue. Each voice-cache identity is warmed at
most once per engine context.

Calls to synthesis, voice changes, and shutdown are serialized. `stop()` sets a
per-request event consumed by the native cancellation callback; the same model
context can be used by a later request. Always call `shutdown()` to cancel work
and release GPU memory deterministically.

## Troubleshooting

Start with:

```bash
python -m qwentts_cpp doctor
```

It reports Python/platform support, driver/GPU information, native ABI, and
discovered CUDA/qwen libraries. `QwenEngine` translates common loader, ABI,
driver, and out-of-memory failures into actionable messages. Typical fixes are:

- ABI mismatch: reinstall matching RealtimeTTS and `qwentts-cpp-python` wheels.
- Missing DLL/SO: inspect `doctor`; do not manually copy random CUDA libraries.
- Driver/CUDA failure: update to a CUDA-12-compatible NVIDIA driver and verify
  compute capability 7.5 or newer.
- VRAM exhaustion: stop other GPU workloads or choose a smaller quantization.
- Offline model error: prefetch the exact model/quant before enabling
  `local_files_only`.

## Reproducible performance benchmark

The benchmark loads the direct binding and RealtimeTTS adapter sequentially,
uses the same text/reference/sampling settings, runs x-vector and ICL, records a
first request plus at least 30 warm requests, and writes representative WAVs:

```bash
python tools/benchmark_qwen_engine.py \
  --ref-audio D:/voices/reference.wav \
  --ref-text "Exact reference transcript." \
  --runs 30 \
  --output-dir D:/Temp/Codex/qwen-benchmark
```

Add `--talker-path` and `--codec-path` together to benchmark direct local GGUF
files rather than Hub-cached files. On a VRAM-constrained system, replace
`--ref-audio` with `--spk-path` and `--rvq-path` to reuse pre-encoded voice
latents and avoid loading the reference encoder during benchmark setup.

`report.json` includes first callback/audio/audible timing, wall time, audio
duration, conventional RTF (wall time divided by audio duration), and native
callback-to-engine-queue overhead. Compare results only when the native
revision, wheel, model files, voice reference, sampling options, and hardware
are the same. Benchmark reports and WAVs are intentionally written outside the
repository.

## Native wheel/release verification

Before publishing a compatible release, build and repair the primary Windows
x86-64 and Linux x86-64 wheels in the `qwentts-cpp-python` project. In fresh
virtual environments on both target operating systems, verify `doctor`, model
loading, x-vector, ICL, repeated requests, cancellation/reuse, and valid 24 kHz
mono output without a repository checkout or system CUDA Toolkit. The current
dev8 candidate deliberately uses the validated platform matrix above; a stable
release still requires final native package versioning and acceptance for both
platforms. Rebuild both artifacts rather than renaming test wheels.
