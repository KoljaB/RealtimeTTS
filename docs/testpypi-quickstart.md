# TestPyPI release candidate: 0.9.0rc7

Use Python 3.11 and uv. Run these commands in the folder where you want your
environment. No activation, generated wrappers, or RealtimeTTS executable
commands are needed. The paths below run the actual installed Python scripts.

## Windows

~~~powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe --find-links https://test.pypi.org/simple/realtimetts/ --find-links https://test.pypi.org/simple/realtimetts-breeze-native/ "realtimetts[breeze-server,qwen-server,qwen-cpu-server,demo]==0.9.0rc7"

# Breeze GPU
.\.venv\Scripts\python.exe .\.venv\Lib\site-packages\RealtimeTTS\breeze_server.py --reference-prefix-cache

# Qwen GPU
.\.venv\Scripts\python.exe .\.venv\Lib\site-packages\RealtimeTTS\qwen_server.py --device gpu

# Qwen CPU (use instead of Qwen GPU, or choose a different --port)
.\.venv\Scripts\python.exe .\.venv\Lib\site-packages\RealtimeTTS\qwen_server.py --device cpu

# Demo, while Breeze and one Qwen server are running
.\.venv\Scripts\python.exe .\.venv\Lib\site-packages\RealtimeTTS\demo.py
~~~

## Linux

~~~sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --find-links https://test.pypi.org/simple/realtimetts/ --find-links https://test.pypi.org/simple/realtimetts-breeze-native/ "realtimetts[breeze-server,qwen-server,qwen-cpu-server,demo]==0.9.0rc7"

# Breeze GPU
.venv/bin/python .venv/lib/python3.11/site-packages/RealtimeTTS/breeze_server.py --reference-prefix-cache

# Qwen GPU
.venv/bin/python .venv/lib/python3.11/site-packages/RealtimeTTS/qwen_server.py --device gpu

# Qwen CPU
.venv/bin/python .venv/lib/python3.11/site-packages/RealtimeTTS/qwen_server.py --device cpu

# Demo
.venv/bin/python .venv/lib/python3.11/site-packages/RealtimeTTS/demo.py
~~~

You can install only the extras you need. Ordinary dependencies come from PyPI;
the two package-specific find-links pages provide the release candidates.
The portable native wheels require Windows/Linux x86-64, a Vulkan driver for
Breeze, and an NVIDIA CUDA 12 compatible driver for Qwen GPU. Linux needs glibc
2.35+. Qwen CPU needs AVX2/FMA/F16C/BMI2.

Breeze Studio: http://127.0.0.1:18084/

Qwen Studio: http://127.0.0.1:8080/

The servers include twelve reference WAVs and their transcripts. The first start
downloads the model files once and prepares the selected voice before readiness.
Local Studio and synthesis require no API key.

## Automatic tuning and manual control

Breeze uses the established first/max chunk settings **4/6** and prints them at
startup. It does not run calibration syntheses or change chunk settings based on
trial requests. These are working defaults, not a claim of optimal performance
on every GPU.

- `--chunk-first` and `--chunk-max` allow explicit manual tuning.
- `--reference-prefix-cache` retains the reference-processing optimization.
- An external `--worker-url` worker keeps its externally configured settings.

Qwen automatically selects the tested profile only when the hardware matches:

| Hardware | Automatic settings |
|---|---|
| Windows Ryzen 9 3900X, verified core layout | 6+6 CPU workers, separate core masks, one-frame chunks, fused attention, 80 ms reserve, AboveNormal process priority |
| Linux i9-13900KF, verified available P-core layout | 7+7 CPU workers, production masks, two-frame chunks, second-chunk startup priority, 80 ms reserve |
| Linux RTX 4090, single unremapped GPU | 40 ms reserve, CUDA graphs disabled, MPS client priority 0 |

Startup logs name the selected profile. Unknown hardware retains portable
settings. Explicit tuning flags and existing environment settings override the
corresponding profile defaults. `--preset portable` disables automatic selection;
`--cpu-core-split` requests manual core allocation. Local `--model`/`--codec` paths
require an explicit compatible preset to avoid assuming their model type. The named presets and
individual CPU worker, affinity, chunk and buffering options remain available;
see [CPU scheduling](qwen-cpu-scheduling.md).

Neither portable native wheel is compiled for these particular machines.
Hardware tuning happens at runtime.

## Speed and live highlighting

Both Studio pages expose Speed and a Words / Characters selector beside the
playback controls. Highlighting starts automatically with WebSocket playback,
using the playback clock and measured timestamps. The multilingual CPU alignment
model is prepared once before readiness, independently of GPU synthesis. It
adds no intentional playback buffer and does not cut or regenerate speech.
Late timestamps cannot be shown retroactively as on-time highlighting.

Speed changes preserve pitch. A speed of 1 bypasses tempo processing; other
speeds need a small processing window. Faster playback also consumes audio
faster, so it requires more synthesis headroom. Alignment timestamps follow the
processed audio. HTTP PCM/WAV also accept speed; live timing uses WebSocket.

Use --no-alignment when starting a server to skip the highlighting model.
The default browser buffer remains 0 ms. The demo retains its own live
highlighting and tagged language/speed controls.
