# Qwen Voice Studio

Studio opens in English, including its default example and messages. Use the
language selector or the DE example button to synthesize German speech.

The Qwen server serves its built-in browser client at `/` and `/studio`. Assets contain no credentials; API and WebSocket inference still require the server's configured key. Enter that key in Connection settings. It stays in the current tab's memory, never local storage. The browser sends WebSocket credentials as a non-echoed subprotocol rather than a query parameter.

Studio streams 24 kHz mono PCM directly into Web Audio as chunks arrive. The default transport is WebSocket; HTTP PCM and complete WAV can also be tested. A user click unlocks audio playback. Pause/resume affects browser playback and sends native checkpoint controls over WebSocket; Stop cancels the request and discards pending playback. In Streaming, open a session, send text fragments, then finish input. WAV download contains the received audio, including a partial recording after Stop.

## Quick start with a cloning example

Create separate environments for CPU and GPU so their dependencies stay clear.
These Windows cmd commands use Python 3.11 and need neither a source checkout
nor an activated environment.

### CPU: Windows Ryzen 9 3900X

```bat
uv venv --python 3.11 --managed-python .venv-cpu
uv pip install --python .venv-cpu\Scripts\python.exe "realtimetts[qwen-cpu-server]==0.8.10"
.venv-cpu\Scripts\realtimetts-qwen-server.exe --preset windows-3900x --cpu-fused-attention --demo-voice
```

The explicit preset is for the tested 12-core/24-thread Ryzen 3900X layout.
It selects six generation and six decoder workers on separate core masks,
AboveNormal process priority, one-frame chunks and an 80 ms server reserve.
The separate `--cpu-fused-attention` flag enables native 0.4.1's optional F32
attention path; the preset alone leaves it off. Explicit CLI values override
the preset. To disable onset recovery, pass
`--onset-silence-profile off --no-onset-silence-recovery`. For other CPUs, use `--device cpu`
and choose worker counts for the hardware; see [CPU scheduling](qwen-cpu-scheduling.md).

### CPU: explicit worker settings on other machines

The minimal `--device cpu --cpu-fused-attention --demo-voice` command still uses
serial decoding. Fused attention alone does not enable the decoder worker pool.
For a machine with 12 physical cores, an overlap starting point is:

```bat
.venv-cpu\Scripts\realtimetts-qwen-server.exe --device cpu --cpu-threads 6 --cpu-codec-threads 6 --cpu-stream-frames 1 --cpu-fused-attention --startup-buffer-ms 80 --demo-voice
```

Adjust both thread counts to the CPU: for example, start with 4+4 on eight
physical cores or 3+3 on six. Count physical cores, not SMT/logical threads, and
leave capacity for other active applications. These are tuning starting points,
not measured performance promises for those CPUs. One stream frame is 80 ms of
audio; the server startup reserve and browser buffer are separate controls.

**Manual 6+6 is not equivalent to `--preset windows-3900x`.** The preset also
sets physical-core masks `0x555` / `0x555000`, AboveNormal process priority,
speaker-only cloning, the checkpoint-specific onset profile
`qwen3_tts_12hz_0_6b_base_q8_v1`, onset recovery and the 80 ms server reserve.
Without explicit options the standard server reserve is 160 ms and the onset
profile is off. Use the preset command above to reproduce the tested 3900X
configuration; do not copy its affinity masks to another CPU topology.

### NVIDIA GPU: Windows

```bat
uv venv --python 3.11 --managed-python .venv-gpu
uv pip install --python .venv-gpu\Scripts\python.exe "realtimetts[qwen-server]==0.8.10"
.venv-gpu\Scripts\realtimetts-qwen-server.exe --device gpu --clone-mode speaker_only --startup-buffer-ms 80 --demo-voice
```

A compatible NVIDIA driver is required. The extra installs the CUDA runtime;
a CUDA Toolkit, Torch and PortAudio are not needed. `--device native` remains
an alias for this GPU path. On Linux, use the corresponding `.venv-gpu/bin/`
executables; the Windows CPU preset is not portable to Linux.

Start one server, then open <http://127.0.0.1:8080/studio> and connect. Select
**demo-neutral** and click **Speak now**. To run both simultaneously,
give the second server `--port 8081` and open its matching Studio URL.

`--demo-voice` downloads only the existing public EARS neutral example, verifies
its pinned SHA-256, registers its reference transcript, and warms the selected
voice before accepting requests. Later starts reuse the saved reference and
encoded voice cache. Initial model download and reference preparation take
longer than a warm start. The example is optional; upload your own recording
under **Voices** for normal cloning. Full ICL remains selectable in Studio.

The first reference download needs network access. `--local-files-only` also
disables that download; it works once the example has been cached or registered.
An existing different voice named `demo-neutral` is preserved and causes a clear
startup error. Choose another `--voice-dir` or omit `--demo-voice` in that case.

Both paths use the same 0.6B Base Q8_0 model by default. No lower-quality model,
quantization or sampling shortcut is selected by these commands. CPU throughput
still depends on contention. Studio defaults to **0 ms additional browser
buffering** for both CPU and GPU. Change **Additional browser buffer (ms)** under
Sampling & options / Codec, transport & more options to add reserve if playback
has gaps, for example 20, 40 or 80 ms. This does not remove the separate server
startup reserve. No finite buffer fixes sustained generation slower than real time.

## Models and supported operations

| Checkpoint | Reference cloning | Fixed speakers | Instructions | Voice design |
|---|---|---|---|---|
| 0.6B Base | Yes | No | No | No |
| 1.7B Base | Yes | No | No | No |
| 0.6B CustomVoice | No | Yes | No | No |
| 1.7B CustomVoice | No | Yes | Yes | No |
| 1.7B VoiceDesign | No | No | Yes | Yes |

Base accepts registered WAV references or native `.spk` plus `.rvq` latents. Speaker-only uses the cached speaker embedding. Full ICL uses reference codes and the exact reference transcript; without a transcript it falls back to speaker-only. Reference extraction happens during registration/preparation, not on every warm synthesis. Base does **not** offer instruction-controlled reference cloning.

CustomVoice lists the actual native speaker names. VoiceDesign uses `instructions` to describe the desired voice; no reference file is required. Unsupported instruction requests return an error instead of silently accepting ineffective controls. Model capability fallback is derived from the variant-qualified model ID/path when the native ABI does not expose model metadata. Use a matching model ID and GGUF file.

The underlying engine and server expose all these modes. Deployments may intentionally load only a subset; the shipped CPU deployment initially remains 0.6B Base after the 1.7B CPU throughput check.

## Loading one or multiple models

Each server process loads exactly its configured `--model-id` / `--model` and codec. There is no browser-triggered download or automatic model eviction/loading. For a single-model setup, start only that server and omit `--studio-peer`; the model selector is disabled.

To offer multiple already-loaded servers in one Studio, start one process per desired checkpoint, on separate ports. On each, list the other running servers with repeatable options such as:

```text
--studio-peer "1.7B Base Q8=http://192.168.178.22:18086"
--cors-origin http://192.168.178.22:18086
```

Configure the reciprocal origin/peer on the other server. Use the same authentication key only for trusted servers in this explicit group. Studio sends its key only to its current origin or an administrator-configured peer selected by the user. Unreachable peers cannot be selected. To run only one model, stop the other processes and remove peer flags; no memory is allocated for unstarted models.

All variants are resolved by the existing qwentts.cpp binding's model catalog. Qwen CPU uses the CPU-only native wheel; this is **not llama.cpp**. The GPU service can remain entirely separate. Model names in requests must match the selected server alias; a mismatched model now returns 400.

## Parameters and measurements

Talker: `seed`, `max_new_tokens`, `do_sample`, `temperature`, `top_k`, `top_p`, `repetition_penalty`. Codec predictor: `subtalker_do_sample`, `subtalker_temperature`, `subtalker_top_k`, `subtalker_top_p`. Additional request fields: `language`, `clone_mode`, `instructions`, `response_format`, `voice`, `model`. Omitted sampling values reset to server defaults for each request. Codec Sampling offers Server-Default, On and Off. With Server-Default and a null backend default, codec sampling inherits the talker setting; it is not forced off. Studio language codes such as `de` and `en` are normalized to the native language names.

The Diagnose tab exposes readiness, health, model information, capability limits, configured defaults and the last request without credentials. Startup-only settings such as CPU threads, model paths, attention mode and silence handling are configuration, not per-request changes.

First PCM is measured at browser receipt, not native generation. Audible onset is estimated from the scheduled first sample with absolute int16 value at least 256, plus the browser's reported output latency. It is not a physical speaker/microphone measurement. RTF includes elapsed time through the last chunk divided by received audio duration; manual fragment waits and pauses are included. The details panel reports scheduled underruns. Engine startup buffer is an amount of audio accumulated, so changing it can cross chunk boundaries and cause stepped latency changes. The independent browser buffer defaults to 0 ms for both CPU and GPU connections, and remains editable. Reported gap duration includes the reserve inserted when playback restarts after an underrun. These values are additional to the server startup reserve; selecting Server-Defaults restores them. The standard server reserve remains 160 ms unless explicitly configured, for example 80 ms in the Windows 3900X launcher.

### Reproducing a latency measurement

Use the exact installed version and launch command above. After startup reports
ready, select **demo-neutral**, English, Speaker-only, the text
`Once when I was six years old, I saw a magnificent picture.` and seed **42**.
Keep the browser buffer at **0 ms**. Use the same Chrome audio output device and
stop competing CPU-heavy work such as video playback. Record the first request
separately, then four warm repetitions, including First PCM, Audible, RTF and
scheduled underruns. The default seed remains random (`-1`); set 42 explicitly
only for this repeatable comparison. Different texts, seeds, voices, system load
or output devices are different measurements.

The prior installed 0.8.9 baseline with these preset settings measured 409-426 ms
estimated audible start, RTF 0.88-0.89 and zero scheduled underruns across four
warm runs at 0 ms additional browser buffer. At 20 ms it measured 430-453 ms and
RTF 0.87-0.89. These are short local measurements, not a p95 or a guarantee under
concurrent workloads. Check the 0.8.10 release notes for the fresh-install results.

No microphone capture is required; references are uploaded from files. Non-WAV formats are decoded and converted by the browser where supported. The API upload size limit applies after base64 encoding. Requests and voice mutations affect the selected server's voice registry only.
