# TestPyPI release candidate: 0.9.0rc4

Use Python 3.11 and [uv](https://docs.astral.sh/uv/). The commands below are the
same on Windows and Linux. Run them from the folder where you want your .venv.
No environment activation or installed RealtimeTTS command wrappers are needed.

## Install

Create the environment once:

~~~sh
uv venv --python 3.11 .venv
~~~

Choose one server installation. You can install several extras into the same
environment. --python .venv explicitly selects that environment for installation.
The package-specific TestPyPI pages supply our RC; regular dependencies come from
PyPI, avoiding unrelated experimental packages on TestPyPI.

Breeze GPU:

~~~sh
uv pip install --python .venv --find-links https://test.pypi.org/simple/realtimetts/ --find-links https://test.pypi.org/simple/realtimetts-breeze-native/ "realtimetts[breeze-server]==0.9.0rc4"
~~~

Qwen NVIDIA GPU:

~~~sh
uv pip install --python .venv --find-links https://test.pypi.org/simple/realtimetts/ "realtimetts[qwen-server]==0.9.0rc4"
~~~

Qwen CPU:

~~~sh
uv pip install --python .venv --find-links https://test.pypi.org/simple/realtimetts/ "realtimetts[qwen-cpu-server]==0.9.0rc4"
~~~

Breeze requires a Vulkan-capable GPU and driver. Qwen GPU requires an NVIDIA
driver compatible with CUDA 12. Native wheels target Windows x86-64 and Linux
x86-64 with glibc 2.35 or newer; x86-64 CPU inference requires AVX2/FMA/F16C/BMI2.
Model weights download on first use and are cached. Breeze model licensing terms
still apply. Server-only installation does not need PortAudio or a sound device.

## Start the Python modules

Each command starts the named Python file from the installed RealtimeTTS package:

~~~sh
uv run --no-project --python .venv python -m RealtimeTTS.breeze_server --reference-prefix-cache
uv run --no-project --python .venv python -m RealtimeTTS.qwen_server --device gpu
uv run --no-project --python .venv python -m RealtimeTTS.qwen_server --device cpu
~~~

Choose the server you need; each runs until stopped with Ctrl+C. --python .venv
selects the same interpreter used for installation. --no-project prevents uv
from discovering or syncing an unrelated project in the current directory.
Python's -m runs breeze_server.py or qwen_server.py inside the installed package,
so there is no separately generated RealtimeTTS executable to find on PATH.

Breeze serves http://127.0.0.1:18084/studio; Qwen serves
http://127.0.0.1:8080/studio. Start only one Qwen instance on that port, or choose
another with --port. Local Studio and synthesis do not require a key.
All twelve reference WAVs and their transcripts are bundled. Each server registers
its own voices automatically. --no-bundled-voices disables that registration.

The Breeze reference-prefix cache reuses reference processing across requests.
The portable native wheel includes it; --reference-prefix-cache selects this
fast path, matching our tuned runtime. Omit the option to compare uncached synthesis.

CPU thread counts and core splitting are optional hardware tuning. The existing
--preset windows-3900x combines that machine's tuned options, not universal defaults.
The --onset-silence-profile option is checkpoint-specific.
See the [measured comparisons and optional CPU startup profile](release-0.9.0rc4-verification.md)
for the settings used to reproduce the tuned Windows/Linux runtimes.

## Tutor demo

The demo uses Breeze for English and Qwen for Spanish. Install both servers and
local audio playback in the same environment (choose qwen-server instead of
qwen-cpu-server if you prefer Qwen on GPU):

~~~sh
uv pip install --python .venv --find-links https://test.pypi.org/simple/realtimetts/ --find-links https://test.pypi.org/simple/realtimetts-breeze-native/ "realtimetts[breeze-server,qwen-cpu-server,demo]==0.9.0rc4"
~~~

On Debian/Ubuntu, local playback also needs PortAudio. If PyAudio needs to build,
install its headers and compiler first:

~~~sh
sudo apt-get install libportaudio2 portaudio19-dev build-essential python3-dev
~~~

Start Breeze and Qwen in separate terminals using the module commands above.
In a third terminal, from the same folder:

~~~sh
uv run --no-project --python .venv python -m RealtimeTTS.demo
~~~

This runs the packaged demo.py: English/Spanish language switching, three speeds,
delivery instructions, laughter, throat clearing, sighs and synchronized terminal
highlighting. It uses the bundled breeze-en and qwen-es voices. No repository
checkout or private reference directory is needed. Its CPU alignment model
downloads once (about 365 MB).

Use --no-highlight for playback without the alignment model, --plain to hide
markup, and --help for options including --breeze-url and --qwen-url.
