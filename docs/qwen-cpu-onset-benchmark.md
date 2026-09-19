# Qwen CPU onset recovery: measured results, 2026-09-13

This is a historical benchmark and private-deployment record, not the current
public installation guide. For RealtimeTTS 0.8.6 and the portable CPU package,
use the [Qwen setup guide](engines/qwen.md#cpu-engine) and
[GPU/CPU installation workflows](qwen-emotions.md).

Conditional recovery reduces long silent starts on the tested CPU setup.
The first native PCM frame is already fast; the avoidable delay is further
generation of silence before usable speech reaches the output queue.

## Final source comparison

54 paired warm requests, four voice/language entries, German and English,
seeds 42, 7 and 314, alternating A/B order. Both arms use the same 0.6B Base
Q8 model, Q8 codec, speaker-only cloning, sampling settings and 180 ms startup
buffer. Hardware: Intel Core i9-13900KF, seven threads, affinity
0,2,6,8,10,12,14. Native revision: `30ea669`.

| Measurement | Existing profile | Conditional recovery |
| --- | ---: | ---: |
| First speech-ready queued PCM, mean | 276.0 ms | 254.2 ms |
| First speech-ready queued PCM, median | 220.9 ms | 217.6 ms |
| First speech-ready queued PCM, p95 | 535.0 ms | 409.4 ms |
| First native PCM, median | 100.5 ms | 100.2 ms |
| ASR word errors / expected words | 5 / 378 | 4 / 378 |
| Correct first words | 54 / 54 | 54 / 54 |

Recovery triggered in four requests. All 50 other requests had identical raw
float PCM and identical queued PCM16 in both arms. No new ASR word errors were
introduced. This supports a reduction in unusually long starts, with little
change to ordinary starts. These are engine measurements, not microphone or
physical speaker timing. p95 describes this sample set, not a service guarantee.

| Recovered request, Mira DE | Seed | Before | After | Trimmed silence before / after |
| --- | ---: | ---: | ---: | ---: |
| Der Test ist abgebrochen. | 42 | 754.9 ms | 419.1 ms | 665 / 275 ms |
| Heute Morgen | 42 | 746.8 ms | 443.3 ms | 690 / 255 ms |
| Heute Morgen | 7 | 527.8 ms | 404.2 ms | 360 / 225 ms |
| Kannst du das bitte noch einmal erklaeren? | 7 | 548.3 ms | 474.1 ms | 380 / 245 ms |

The after column includes all time spent on the abandoned first attempt. The
trimmed-silence after column likewise includes the initial 240 ms quiet probe;
the retry itself does not generate that initial probe again.

## Mechanism and rejected alternatives

The existing v1 profile already applies on CPU. Extending its suppression from
three to six codec frames did not consistently improve onset. Adding c0 ID
1221 gave much larger immediate gains, but sometimes introduced an extra
initial "Ha". That defect reproduced with both one-frame and three-frame
masks. Adding 2042 also introduced an extra syllable and made other starts
substantially slower. Neither blanket mask is accepted.

The selected implementation observes the ordinary stream. Only after 240 ms
of entirely quiet generated PCM does it close that attempt and retry once
with the CPU-specific v2 profile (v1 IDs plus 1221). It checks the entire
received boundary chunk for speech, verifies that the old native producer
has stopped, preserves cancellation, and records both attempts separately.
The recovery consumer never acknowledges native pause checkpoints. A pause
already requested at the retry boundary keeps the original stream alive;
the pause check and producer close are serialized to avoid a paused join.
The exact talker/codec hash guards remain enforced. ICL and trim-disabled
requests do not retry. The option is off by default.

Removing the startup buffer made the initial queue faster but produced
predicted underruns in all eight control runs, so that change was rejected.

## Continuity and content checks

The final comparison had 3 versus 7 runs with predicted underruns, all on
ordinary, nonretry requests. A previous 54-pair comparison had 9 versus 8.
A targeted direct-engine repeat on the seven affected ordinary cases, plus
the four recovered starts, had 5 versus 4 in 25 pairs. All 50 repeat outputs
matched the already checked PCM exactly. The four recovered starts had no
predicted underruns in these comparisons. The evidence does not show a
systematic recovery overhead; it does show that the existing 180 ms buffer
has little margin against occasional native timing jitter. Continuous
physical playback was not measured.

ASR used the explicitly approved local Parakeet v3 INT8 service, with ffmpeg
resampling to 16 kHz and no expected-transcript prompt. ASR checks content;
it does not establish voice identity or prosody equivalence.

An initial integration benchmark was rejected because its recorder did not
forward generator close, making the retry wait for the original synthesis.
The final recorder explicitly closes the wrapped stream. Direct-engine
repeats independently reproduce the gain: 766.6 -> 424.8 ms and
808.0 -> 410.5 ms on the two worst starts.

## Reproduction and code

- Engine option: `QwenCpuEngine(onset_silence_recovery=True, onset_silence_profile="qwen3_tts_12hz_0_6b_base_q8_v1")`.
- Server option: `--device cpu --onset-silence-recovery`, with the v1 profile.
- Required CPU binding: `realtimetts-qwen-native==0.2.0+cpu2`.
- Local evidence: `test_outputs/qwen_cpu_onset_20260913/`, including exact
  experiment scripts, JSON, ASR responses, source snapshots, rejected runs,
  native patch, and SHA-256 manifests.
- Engine/server regression checks: 96 passed, one local native-package skip.
  Native profile and stream-close checks also passed. The updated CPU package
  dependency guard passed its focused check.

The deployed package is built from the prior live source commit plus the
onset changes. Existing unrelated local work is preserved separately.

## Verified Linux deployment

Deployed to `wwz-qwen3-tts-cpu.service` on `192.168.178.22:18085` with
recovery enabled. Healthy PID: `44088`. Installed versions:
`realtimetts==0.8.6+studio2.onset2` and
`realtimetts-qwen-native==0.2.0+cpu2`.
The model, codec, CPU threads, affinity, and startup buffer remain those
used in the comparison above.

The final wheels reproduced all ten selected benchmark PCM outputs exactly.
Twelve authenticated HTTP requests before and twelve after deployment then
exercised all seven configured language routes. The German rows are means
of two requests each; the other rows are single requests. These sequential
deployment checks confirm operation and the slow-start gain; they are too
small to estimate a new p50/p95 or attribute ordinary timing fluctuations.

| HTTP first usable PCM | Before | After |
| --- | ---: | ---: |
| DE: Der Test ist abgebrochen. | 763.5 ms | 423.8 ms |
| DE: Heute Morgen | 748.3 ms | 417.3 ms |
| DE: Kannst du das bitte noch einmal erklaeren? | 325.2 ms | 327.4 ms |
| EN | 205.6 ms | 227.5 ms |
| ES | 214.1 ms | 218.2 ms |
| FR | 207.1 ms | 209.9 ms |
| IT | 546.3 ms | 446.1 ms |
| PT | 212.5 ms | 214.6 ms |
| RU | 276.7 ms | 210.4 ms |

Local ASR found 2 versus 1 word errors in 62 expected words per arm, no new
word errors, and every initial word present. The remaining French gender
ending difference was already present before deployment. All seven ordinary
request outputs matched the old service PCM exactly. Installed-package tests
also passed native pause/resume, cancellation of a paused retry, and a fresh
synthesis after cancellation.

Both components passed signed `release_guard.py attest` and `verify`, with
exact source/wheel/sdist/runtime parity and a rechecked live service.
Source commits: RealtimeTTS `511922c7abaf81b416af989b4446fdee0d0841d4`;
native binding `6c7d7f2d795e1fd62c0a90ac597cd2d8e6d23803`.
The CPU2 wheel reuses the exact tested CPU1 native library bytes; only the
Python binding changed. This was a private deployment, with no public release.

Deployment evidence and detached signatures are under
`test_outputs/qwen_cpu_onset_20260913/deployment/` locally and
`/home/lon/Dev/qwen-cpu-onset-20260913/realtimetts/` on Linux. Exact previous
wheels and the original service unit remain available for rollback.
An initial service switch failed because the deployment script inserted two
flags inside one quoted argument. The automatic rollback restored the old
packages; the corrected switch validated the complete argument list with the
new wheel's parser before restarting successfully.
