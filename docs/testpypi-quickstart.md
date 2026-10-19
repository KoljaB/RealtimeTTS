# TestPyPI release candidate: 0.9.0rc3

Windows PowerShell, Python 3.11, with [uv](https://docs.astral.sh/uv/).
Create one environment; choose any of the three server extras below (they may also
share an environment). Activate it so all following commands are short:

~~~powershell
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
~~~

Install the desired server. TestPyPI supplies the candidate; PyPI supplies its
normal dependencies. The version pin selects this exact RC.

~~~powershell
uv pip install --index https://test.pypi.org/simple --index-strategy unsafe-best-match "realtimetts[breeze-server,demo]==0.9.0rc3"
# Or Qwen on NVIDIA GPU:
uv pip install --index https://test.pypi.org/simple --index-strategy unsafe-best-match "realtimetts[qwen-server,demo]==0.9.0rc3"
# Or Qwen on CPU:
uv pip install --index https://test.pypi.org/simple --index-strategy unsafe-best-match "realtimetts[qwen-cpu-server,demo]==0.9.0rc3"
~~~

Start the selected server:

~~~powershell
realtimetts-breeze-server
realtimetts-qwen-server --device gpu
realtimetts-qwen-server --device cpu
~~~

Breeze listens on port 18084; Qwen on 8080. Run only one Qwen instance on that
port, or choose another with --port. Model weights download on first use and
are cached. Breeze uses Vulkan and needs a compatible GPU driver; Qwen GPU uses
CUDA 12 with an NVIDIA GPU. Breeze's model licensing terms still apply.
All twelve reference WAVs and their transcripts are included in the package.
The servers register their engine's references automatically, without extracting
all voice embeddings at startup. --no-bundled-voices disables this.

The CPU defaults are portable. Thread counts, separate codec workers, affinity
masks and CPU core splitting are optional hardware tuning. The existing
--preset windows-3900x combines that machine's tuned options; it is not a
universal recommendation. --onset-silence-profile is checkpoint-specific.

## Tutor demo

Install both Breeze and either Qwen server, then start them in separate terminals
with the same environment activated. In a third terminal:

~~~powershell
realtimetts-demo
~~~

This is the packaged version of the English/Spanish tutor example: language
switching, three speeds, delivery instructions, laughter, throat clearing, sighs
and synchronized terminal highlighting. It uses breeze-en and qwen-es from the
bundled references. Its multilingual CPU alignment model downloads once (about
365 MB). No repository checkout or private reference directory is required.

Use --no-highlight for playback without the alignment download. --plain hides
markup in the highlight display. --breeze-url, --qwen-url and --alignment-model
override the default endpoints or cached alignment model. Run realtimetts-demo
--help for the full list. On Linux activate with source .venv/bin/activate;
local playback also requires PortAudio system libraries.
