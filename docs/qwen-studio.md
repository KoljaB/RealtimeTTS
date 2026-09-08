# Qwen Voice Studio

The Qwen server serves its built-in browser client at `/` and `/studio`. Assets contain no credentials; API and WebSocket inference still require the server's configured key. Enter that key in Connection settings. It stays in the current tab's memory, never local storage. The browser sends WebSocket credentials as a non-echoed subprotocol rather than a query parameter.

Studio streams 24 kHz mono PCM directly into Web Audio as chunks arrive. The default transport is WebSocket; HTTP PCM and complete WAV can also be tested. A user click unlocks audio playback. Pause/resume affects browser playback and sends native checkpoint controls over WebSocket; Stop cancels the request and discards pending playback. In Streaming, open a session, send text fragments, then finish input. WAV download contains the received audio, including a partial recording after Stop.

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

Talker: `seed`, `max_new_tokens`, `do_sample`, `temperature`, `top_k`, `top_p`, `repetition_penalty`. Codec predictor: `subtalker_do_sample`, `subtalker_temperature`, `subtalker_top_k`, `subtalker_top_p`. Additional request fields: `language`, `clone_mode`, `instructions`, `response_format`, `voice`, `model`. Omitted sampling values reset to server defaults for each request.

The Diagnose tab exposes readiness, health, model information, capability limits, configured defaults and the last request without credentials. Startup-only settings such as CPU threads, model paths, attention mode and silence handling are configuration, not per-request changes.

First PCM is measured at browser receipt, not native generation. Audible onset is estimated from the scheduled first sample with absolute int16 value at least 256, plus the browser's reported output latency. It is not a physical speaker/microphone measurement. RTF includes elapsed time through the last chunk divided by received audio duration; manual fragment waits and pauses are included. The details panel reports scheduled underruns. Engine startup buffer is an amount of audio accumulated, so changing it can cross chunk boundaries and cause stepped latency changes. The independent browser buffer defaults to 8 ms and can be set to zero.

No microphone capture is required; references are uploaded from files. Non-WAV formats are decoded and converted by the browser where supported. The API upload size limit applies after base64 encoding. Requests and voice mutations affect the selected server's voice registry only.
