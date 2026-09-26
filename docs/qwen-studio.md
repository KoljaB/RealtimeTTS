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
uv venv --python 3.11 .venv-cpu
uv pip install --python .venv-cpu\Scripts\python.exe "realtimetts[qwen-cpu-server]==0.8.9"
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

### NVIDIA GPU: Windows

```bat
uv venv --python 3.11 .venv-gpu
uv pip install --python .venv-gpu\Scripts\python.exe "realtimetts[qwen-server]==0.8.9"
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
still depends on contention: the 80 ms server reserve and Studio's separate
80 ms CPU browser buffer cannot hide sustained generation slower than real time.
GPU Studio also uses an 80 ms browser buffer to cover startup scheduling jitter.

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

First PCM is measured at browser receipt, not native generation. Audible onset is estimated from the scheduled first sample with absolute int16 value at least 256, plus the browser's reported output latency. It is not a physical speaker/microphone measurement. RTF includes elapsed time through the last chunk divided by received audio duration; manual fragment waits and pauses are included. The details panel reports scheduled underruns. Engine startup buffer is an amount of audio accumulated, so changing it can cross chunk boundaries and cause stepped latency changes. The independent browser buffer defaults to 80 ms for both CPU and GPU connections, and remains editable. Reported gap duration includes the reserve inserted when playback restarts after an underrun. These values are additional to the server startup reserve; selecting Server-Defaults restores them. The standard server reserve remains 160 ms unless explicitly configured, for example 80 ms in the Windows 3900X launcher.

No microphone capture is required; references are uploaded from files. Non-WAV formats are decoded and converted by the browser where supported. The API upload size limit applies after base64 encoding. Requests and voice mutations affect the selected server's voice registry only.
