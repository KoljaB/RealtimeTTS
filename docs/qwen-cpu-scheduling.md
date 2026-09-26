# Qwen CPU scheduling

RealtimeTTS 0.8.9 pins realtimetts-qwen-native-cpu 0.4.0 for the CPU extras.
The CUDA native package remains at 0.2.0. Model weights, Q8_0 precision, voices
and sampling settings are unchanged by the CPU scheduling options.

Intel Mac installations select Numba 0.62.x for prebuilt dependencies; use
Python 3.10-3.13 on that platform. Apple Silicon uses its ordinary dependency
selection. See the [Numba support change](https://numba.readthedocs.io/en/latest/release/0.63.0-notes.html).

## Enable overlap

Choose worker counts for the actual machine. This example allocates four
workers to code generation and four to decoding; it does not detect topology.

```python
from RealtimeTTS import QwenCpuEngine

engine = QwenCpuEngine(cpu_threads=4, cpu_codec_threads=4, cpu_stream_frames=2)
```

Equivalent server options:

```text
--device cpu --cpu-threads 4 --cpu-codec-threads 4 --cpu-stream-frames 2
```

cpu_codec_threads=0 preserves serial decoding. A positive value creates the
independent decoder pool. cpu_stream_frames accepts 0 (native default), 1, 2
or 4. At 12.5 acoustic frames/s, these correspond to 80/160/320 ms steady-state
chunks. This is separate from startup_buffer_ms, which remains 160 by default.
Overlap supports one utterance per context (max_batch=1).

Optional cpu_affinity and cpu_codec_affinity are unsigned 64-bit logical-CPU
masks, defaulting to zero (unpinned). CLI names use hyphens. Check topology
before choosing masks: physical cores and SMT siblings share resources. Do
not copy a benchmark host's mask to another machine. Unsupported or unavailable
affinity layouts must be checked on the target platform.


## Windows Ryzen 9 3900X: ready-to-run server

For the tested Windows Ryzen 9 3900X (12 physical cores, 24 logical CPUs), use
[server_3900x.cmd](https://github.com/KoljaB/RealtimeTTS/blob/v0.8.9/tools/qwen_cpu_overlap/server_3900x.cmd). This is an explicit
machine-specific profile, not automatic CPU detection. Do not use these affinity
masks on a different processor or logical-core layout.

In Windows cmd, create a folder for the test and run:

```bat
uv venv --python 3.11
uv pip install --python .venv\Scripts\python.exe "realtimetts[qwen-cpu-server]==0.8.9"
.venv\Scripts\realtimetts-qwen-server.exe --preset windows-3900x --demo-voice
```

The profile now lives in the installed package. Downloading the CMD launcher is
optional; it calls the same profile and enables the demo voice. The public neutral
reference is registered and warmed before the server is ready. For a matching
GPU installation, see the [CPU/GPU quick start](qwen-studio.md#quick-start-with-a-cloning-example).

The launcher uses an activated venv, a `.venv` next to the downloaded file, or
the repository-root `.venv` when run from a checkout. It uses the normal model
cache and permits the first-run model download. It does not require PowerShell.
Keep the server running and open http://127.0.0.1:8080/studio. Connect without a
key for the default local-only server and select **demo-neutral**, or add your
own reference under **Stimmen**. An optional transcript is needed for Full ICL; speaker-only cloning
does not require one.

The optional emotional demo can also register and synthesize the neutral
reference through an existing server:

```bat
.venv\Scripts\python.exe -m RealtimeTTS.qwen_emotions --server http://127.0.0.1:8080 --emotions neutral --no-play
```

The launcher selects six generation workers and six codec workers, one-frame
chunks, and separate physical-core masks `0x555` / `0x555000`. It sets only its
own Python process to Windows **AboveNormal** priority; it does not change
system-wide settings. The server startup reserve is 80 ms. Studio adds its
separate, editable 80 ms CPU browser buffer. Extra CLI arguments can be appended,
for example `server_3900x.cmd --port 8081`.

Windows scheduling and competing workloads can change throughput. Earlier Ryzen
3900X browser checks with a two-frame cadence completed short German streams at displayed
RTF 0.95 and 0.93 with zero scheduled buffer gaps; the corresponding estimated
audible starts were 596 and 626 ms. These are local measurements, not a guarantee
for other texts or systems. Later release checks under concurrent CPU load also
recorded gaps at RTF 1.11-1.13. The one-frame preset removes the abrupt change
from an 80 ms first chunk to 160 ms steady chunks: a seeded paired check kept
identical PCM and removed a 41 ms initial arrival gap at the same buffer size.
Without the preset, launcher or explicit worker
options, the existing server scheduling defaults remain serial and unpinned.

## Optional startup priority

Set QWENTTS_CPU_STARTUP_PRIORITY=second_chunk before creating the CPU engine
to prioritize the second native decode block, then resume normal overlap.
Unset it or set off to retain normal overlap. It requires a positive codec
worker count and a two/four-frame cadence (or native default); incompatible
settings fail before models load. GPU initialization ignores this CPU option.

The Linux comparison recovered 18-23 ms of first audible audio, with a measured
2-6% generation-time cost. Keep this an explicit latency preference.

## Evidence and limits

On an i9-13900KF, paired Linux synthesis with real Windows Miep playback reduced
RTF from 0.622-0.635 to 0.471-0.487 at the same 160 ms startup reserve: 23-26%
lower RTF, or 30-35% greater throughput. The three text/voice outputs and seven
language-route checks matched the prior PCM exactly. This measures those
inputs and configurations, not every processor or workload.

The subsequent 80 ms setup is a deployment choice, not a portable default or a
guarantee against underruns. One older 195 ms empty-buffer event remains
unexplained; later checks did not reproduce it. Timing of audible audio used
PortAudio's first non-silent sample DAC schedule, not a microphone measurement.

GPU finer-startup scheduling remains experimental and is excluded here.
The complete emotional demo remains editable in tests/faster_qwen_emotions.py;
compact colored output is the default and --verbose exposes native diagnostics.
