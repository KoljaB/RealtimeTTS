# Qwen CPU scheduling

RealtimeTTS 0.8.7 pins realtimetts-qwen-native-cpu 0.4.0 for the CPU extras.
The CUDA native package remains at 0.2.0. Model weights, Q8_0 precision, voices
and sampling settings are unchanged by the CPU scheduling options.

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
