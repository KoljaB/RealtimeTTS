# RC4 portable build and runtime verification — 2026-10-19

Correction for RC7: the Breeze comparison below normalized both binaries to
1/40 chunks. It did not establish parity with the actual working 4/6 launcher.
The simple CPU/GPU launches also omitted hardware tuning used by the reference
services. RC7 restores the verified profiles and adds measured Breeze startup
selection; see the [current quickstart](testpypi-quickstart.md). The historical
numbers below must not be presented as evidence of complete launcher parity.

RealtimeTTS 0.9.0rc4 uses realtimetts-breeze-native 0.1.0rc2,
realtimetts-qwen-native 0.2.0 and realtimetts-qwen-native-cpu 0.4.2.
See [installation and Python module commands](testpypi-quickstart.md).

## Portable builds and optional tuning

The Breeze native source patch now includes the working reference-prefix cache,
bounded preceding context, invalidation and timing metrics. Both OS builds use
GGML_NATIVE=OFF. The Windows build uses pinned Winlibs GCC 12.2.0 and MSYS2
shaderc 2026.3 / glslang 16.3.0 / Vulkan and SPIR-V headers 1.4.357.0. All downloaded
archives have fixed SHA-256 checksums. This matches the working Windows compiler
and shader tools without compiling for the build host's CPU. Linux uses the
pinned LunarG 1.3.296.0 SDK and an Ubuntu 22.04 build repaired by auditwheel.

The native build is maintained in
[realtimetts-breeze-native](https://github.com/KoljaB/realtimetts-breeze-native/tree/codex/breeze-portable-cache).
[The final Windows/Linux CI build passed](https://github.com/KoljaB/realtimetts-breeze-native/actions/runs/39257706183).
The verified Linux wheel came from the preceding commit with the same Linux build
inputs; subsequent changes only completed the Windows SDK installation.

The native wheels target Windows x86-64 and Linux x86-64 with glibc 2.35+.
Breeze requires a Vulkan driver; Qwen GPU requires a suitable NVIDIA CUDA 12
driver. Qwen x86-64 CPU inference requires AVX2/FMA/F16C/BMI2. Thread counts,
affinity, startup buffering and CPU startup priority remain runtime choices.
No local affinity masks or GPU device numbers are embedded in the wheel.

## Measured comparisons

These are bounded warm comparisons, not guarantees for every input, machine or
competing workload. The Windows GPU also drives the desktop. Cold model loading,
reference preparation and shader compilation are excluded from warm medians.
A first PCM packet is not the same as sound reaching a listener's speaker.

| Comparison | Working runtime | RC4 candidate | Evidence |
|---|---:|---:|---|
| Windows Breeze, first PCM at Python consumer | 179.21 ms | 181.96 ms | Six warm runs each, A-B-B-A order |
| Windows Breeze, RTF | 0.895 | 0.919 | Same 1.84 s output duration; reference cache hits |
| Linux Qwen GPU, first PCM over HTTP | 70.8 ms | 66.6 ms | Three paired warm runs; identical PCM 3/3 |
| Linux Qwen GPU, synthesis wall time | 294.0 ms | 288.9 ms | Same inputs and sampling |
| Linux Qwen CPU, native synthesis to first PCM | 170.978 ms | 172.786 ms | Three runs each with the same startup-priority setting |
| Linux Qwen CPU, first WebSocket binary packet | 180.293 ms | 186.976 ms | Identical PCM hashes and byte counts |
| Linux Qwen CPU, synthesis wall time | 2378.989 ms | 2077.877 ms | Same CPU pools, affinity and sampling |

Breeze used an RTX 2080 SUPER, Q8_0 model, bundled English voice, seed 44,
max_new_tokens 160, chunk-first 1, chunk-max 40, reference-prefix caching and
"Hello, this is a short test." Each of the four process blocks had one discarded
warmup and three measured requests. The earlier MSVC/older-shader-tool build was
slower and was rejected; its measurements are not the release result.

Linux Qwen used the existing RTX 4090 GPU and i9-13900KF CPU reference runtimes,
the same voice/model/input per pair, seed 42 and identical sampling. The initial
CPU comparison omitted a production environment setting and was not equivalent.
With that setting restored, the native first-PCM difference fell from 29.040 ms
to 1.808 ms. The optional alignment dependencies were also installed to reproduce
the production profile. Alignment is not required for basic TTS serving.

Linux Breeze additionally produced real PCM with a working cache: warm median
first PCM 139.09 ms, RTF 0.526. No earlier Linux Breeze performance baseline was
available. Windows Qwen CPU/GPU received real synthesis checks, not paired
performance claims. These checks do not establish new hallucination rates.

## Reproduce the tuned CPU startup profile

The simple CPU command in the quickstart works without this tuning. To reproduce
the measured i9-13900KF configuration, create a text file named qwen-cpu.env:

~~~dotenv
QWENTTS_CPU_STARTUP_PRIORITY=second_chunk
~~~

Then use the same explicit interpreter selection on Windows or Linux:

~~~sh
uv run --no-project --python .venv --env-file qwen-cpu.env python -m RealtimeTTS.qwen_server --device cpu --cpu-threads 7 --cpu-codec-threads 7 --cpu-stream-frames 2 --cpu-affinity 0x5545 --cpu-codec-affinity 0xaa8a --startup-buffer-ms 80 --onset-silence-profile qwen3_tts_12hz_0_6b_base_q8_v1 --clone-mode speaker_only
~~~

Those counts and affinity masks describe that machine; select suitable values
on other processors. The startup preference requires codec overlap and a
compatible two/four-frame cadence. Do not combine it with the one-frame
windows-3900x preset. See [CPU scheduling](qwen-cpu-scheduling.md#optional-startup-priority).
When enabling the existing production alignment features, also install
alignment-cpu and alignment-multilingual extras and supply the alignment models;
the CPU server extra alone intentionally omits those optional dependencies.

## Installation acceptance

Both operating systems resolve the documented directory form
uv run --no-project --python .venv python to that environment's interpreter.
The installed Breeze and Qwen modules serve real audio and bundled voices.
Studio and synthesis work without a key in the default local configuration.
The Windows Breeze module check took 108.75 s to become ready, then synthesized
1.84 s of audio in 1.59 s. The Linux module became ready in 6.53 s. Startup
was not a paired benchmark; the warm latency figures above exclude startup.
The packaged demo is available as python -m RealtimeTTS.demo; twelve reference
WAVs and transcripts are bundled. Local playback on Linux requires PortAudio.

The Linux production CPU service remained unchanged. Temporary test servers
were stopped, and the Linux LLM remained off as requested.

Verified native artifact SHA-256 values:

- Windows: 692e848071bf7bd286b9f41662553e0c314deefa49458430b086fcf0481423a5
- Linux: 0745ff4d319c989fb65d7007822fae2d93beb0dd3647f58c64240c9f3bc9031b
