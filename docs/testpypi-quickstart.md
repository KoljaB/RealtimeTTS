# TestPyPI release candidate: 0.9.0rc6

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
uv pip install --python .venv --find-links https://test.pypi.org/simple/realtimetts/ --find-links https://test.pypi.org/simple/realtimetts-breeze-native/ "realtimetts[breeze-server]==0.9.0rc6"
~~~

Qwen NVIDIA GPU:

~~~sh
uv pip install --python .venv --find-links https://test.pypi.org/simple/realtimetts/ "realtimetts[qwen-server]==0.9.0rc6"
~~~

Qwen CPU:

~~~sh
uv pip install --python .venv --find-links https://test.pypi.org/simple/realtimetts/ "realtimetts[qwen-cpu-server]==0.9.0rc6"
~~~

Breeze requires a Vulkan-capable GPU and driver. Qwen GPU requires an NVIDIA
driver compatible with CUDA 12. Native wheels target Windows x86-64 and Linux
x86-64 with glibc 2.35 or newer; x86-64 CPU inference requires AVX2/FMA/F16C/BMI2.
Model weights download on first use and are cached. Breeze model licensing terms
still apply. Server-only installation does not need PortAudio or a sound device.

## Start the Python files

Run the desired command from the folder containing .venv. These are the full
server implementations installed by the package, not additional launcher files.

Windows (PowerShell or cmd):

~~~sh
.venv\Scripts\python.exe .venv\Lib\site-packages\RealtimeTTS\breeze_server.py --reference-prefix-cache
.venv\Scripts\python.exe .venv\Lib\site-packages\RealtimeTTS\qwen_server.py --device gpu
.venv\Scripts\python.exe .venv\Lib\site-packages\RealtimeTTS\qwen_server.py --device cpu
~~~

Linux (with the Python 3.11 environment created above):

~~~sh
.venv/bin/python .venv/lib/python3.11/site-packages/RealtimeTTS/breeze_server.py --reference-prefix-cache
.venv/bin/python .venv/lib/python3.11/site-packages/RealtimeTTS/qwen_server.py --device gpu
.venv/bin/python .venv/lib/python3.11/site-packages/RealtimeTTS/qwen_server.py --device cpu
~~~

Choose one Qwen variant. Each server runs until stopped with Ctrl+C.
For shorter commands or editing, you can copy these complete .py files into
your working folder and run the same environment Python with breeze_server.py,
qwen_server.py or demo.py. Their imports use the installed RealtimeTTS package.

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
uv pip install --python .venv --find-links https://test.pypi.org/simple/realtimetts/ --find-links https://test.pypi.org/simple/realtimetts-breeze-native/ "realtimetts[breeze-server,qwen-cpu-server,demo]==0.9.0rc6"
~~~

On Debian/Ubuntu, local playback also needs PortAudio. If PyAudio needs to build,
install its headers and compiler first:

~~~sh
sudo apt-get install libportaudio2 portaudio19-dev build-essential python3-dev
~~~

Start Breeze and Qwen in separate terminals using the script commands above.
In a third terminal, from the same folder:

Windows:

~~~sh
.venv\Scripts\python.exe .venv\Lib\site-packages\RealtimeTTS\demo.py
~~~

Linux:

~~~sh
.venv/bin/python .venv/lib/python3.11/site-packages/RealtimeTTS/demo.py
~~~

This runs the packaged demo.py: English/Spanish language switching, three speeds,
delivery instructions, laughter, throat clearing, sighs and synchronized terminal
highlighting. It uses the bundled breeze-en and qwen-es voices. No repository
checkout or private reference directory is needed. Its CPU alignment model
downloads once (about 365 MB).

Use --no-highlight for playback without the alignment model, --plain to hide
markup, and --help for options including --breeze-url and --qwen-url.
