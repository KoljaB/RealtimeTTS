# realtimetts-breeze-native

This directory contains a small prebuilt-native wheel assembler for the
RealtimeTTS Breeze worker. The TestPyPI candidate is 0.1.0rc2.
Windows and Linux builds are produced by GitHub Actions in
https://github.com/KoljaB/realtimetts-breeze-native.

The wheel contains the pinned patched breeze-server executable, a shell-free
launcher, third-party notices, and deterministic build-info.json. Model GGUF
files are supplied separately; the runtime reads tokenizer and vocoder data
from the selected GGUF. The wheel has no downloader or cache manager.

## Pinned inputs and model scope

- Breeze-TTS-2.cpp: a0e177f91242ebbd2ea8d61548d742c89e4b9e06
- ggml: 36da57138425487184aa1da2eee2cde155909c6f
- control patch SHA-256:
  5d4e5e3b69c0fa1f81bb4d291b78764270c6dbb18656e2a4ce4bce9f2dbabf4e

The first release targets the official Q8_0 model only. CFG=4 remains the
tested default. CFG=1 can be documented later with its quality tradeoff.
Experimental dd variants are outside this package contract.

## Licensing

The Breeze TTS 2 model and its self-hosted outputs are for research and
non-commercial use only. Commercial use requires a separate licence from
BreezeBlue. Read the [BreezeBlue model licence](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE)
and contact [contact@breeze.blue](mailto:contact@breeze.blue).

## Build contract

The assembler accepts only raw linux_x86_64 and win_amd64 builds. It checks
the pinned source and ggml revisions, installed controls, CMake source
directory, Release mode, static/Vulkan/MP3/CUDA/native flags, and one
non-empty worker. build-info.json records those checks and deliberately
omits absolute compiler and Vulkan tool paths.

Required cache values:

- BREEZE_BUILD_CLI=OFF
- BREEZE_BUILD_SERVER=ON
- BREEZE_BUILD_SHARED=OFF
- BREEZE_BUILD_TESTS=OFF
- BREEZE_CUDA=OFF
- BREEZE_MP3=OFF
- BREEZE_VULKAN=ON
- BUILD_SHARED_LIBS=OFF
- GGML_CUDA=OFF
- GGML_NATIVE=OFF
- GGML_VULKAN=ON

GGML_NATIVE=OFF is intentional for the initial x64 target. Vulkan loader
and driver support remain host runtime requirements.

## Local Windows assembly

Run from the repository root. Replace angle-bracket placeholders with paths
outside this checkout:

~~~
$env:BREEZE_SOURCE_DIR = "<external-source-dir>"
$env:BREEZE_BUILD_DIR = "<release-build-dir>"
$env:BREEZE_NATIVE_STAGE_DIR = "<empty-stage-dir>"
$env:BREEZE_WHEEL_PLATFORM = "win_amd64"
$env:BREEZE_WHEELHOUSE = "<wheelhouse-dir>"
python -B native\breeze\build_package.py
python -m build --wheel --outdir $env:BREEZE_WHEELHOUSE native\breeze
~~~

The build script always uses the repository's
tools/breeze_cpp_controls.json and its patch checker. The stage directory
must be empty and outside the repository.

## Local Linux assembly

Use BREEZE_WHEEL_PLATFORM=linux_x86_64 with the same command. The result is a
raw linux_x86_64 wheel. Inspect that artifact with auditwheel show, then run
auditwheel repair --plat manylinux_2_35_x86_64 only after reviewing its Vulkan
and system-library dependencies. Auditwheel, not this assembler, produces the
manylinux tag.

The assembler does not vendor Vulkan loaders, GPU drivers, glibc, or model
weights. Record the repaired wheel hash and auditwheel output in release
evidence before any native publication.

## Runtime boundary

The launcher only exposes the packaged executable and CLI argument list.
RealtimeTTS owns loopback worker lifecycle, readiness, cancellation, and
cleanup. This package leaves the existing external-worker path unchanged.
