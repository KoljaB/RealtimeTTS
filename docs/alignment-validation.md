# Alignment validation: 9-10 October 2026

The concrete coverage, numeric projection, repeated-work, MMS API and
multilingual word-boundary defects are fixed. Completion now requires explicit
acoustic evidence, and Studio can apply a verified cutoff to audio still queued
for playback. The measurements below also show the limits: difficult/noisy speech
can remain unverified, and a missed endpoint can leave surplus speech intact.

## Follow-up: 10 October 2026

The browser playback and fresh evaluation runs are complete. No production
service was deployed or restarted.

Uncensored capture covered 24 development utterances / 82.100 seconds across
seven languages. It exposed an audio-origin error when the first Portuguese word
moved from 80 ms to 0 ms, a server-only import of PyAudio, and verified cutoff
events being swallowed when alignment display was requested. These are fixed.
Four difficult English recordings still have insufficient acoustic support;
confidence thresholds were not relaxed to force them to pass.

### Actual browser playback

Seven controlled cases passed in Chrome using the real AudioContext and
AudioWorklet: intact audio, repeat leakage, queued and late cutoffs, stale
metadata, and two following-fragment cases. A real 3,075 ms scheduling gap after
trimming a queued fragment is fixed; both following-fragment gaps are now 0 ms.
All intended PCM was preserved. A late cutoff still allowed 68.958 ms of
above-threshold surplus signal that had already played.

Twelve native WebSocket requests also produced browser audio. Comparing raw
render blocks in invocation order with scheduled PCM found zero missing or
mismatched samples before the declared final fade, in all 12 cases. Eight
captures had inconsistent AudioWorklet frame labels, so their timestamp-indexed
captures and precise clock measurements are excluded. One scheduled underrun
was 5.333 ms. These are browser output checks, not acoustic loopback measurements.

### Held-out failure, correction, and final fresh evaluation

The first frozen candidate was evaluated on 24 fresh utterances / 65.385 seconds
plus eight compound-prefix negatives. It falsely verified all eight negatives;
one natural recording lost 415 ms. That failed result is retained.

The correction requires support for the final spoken character and rejects
confident continuing letters after the proposed endpoint. Stability, a word
separator, and end-of-input cannot override that continuation evidence. On the
same set, now a regression test, false positives fell to 0/8 and no natural
recording was shortened. Completion verified for 18/24 natural recordings and
21/24 controlled repeats.

A final candidate was then frozen before generating another **16 fresh
utterances / 39.410 seconds**, with six new compound-prefix negatives. Texts and
seeds were predeclared; speakers were reused. There was no tuning on this set:

| Final fresh set | Verified endings | False prefix completions |
| --- | ---: | ---: |
| Natural recordings | 14/16 | 0/6 |
| Repeat appended after 300 ms | 15/16 | 0/6 |

All 22 natural replays retained their original duration and matched unaligned
PCM sample for sample before the final 25 ms fade. There were no errors. The two
unverified natural endings stayed intact. A verified endpoint still may arrive
late; these counts do not establish prevention of every surplus sample or broad
word-boundary accuracy.

### Difficult Breeze voices

The supplied squeaky mouse, gravelly dragon and croaking/bubbling sprite recordings
provided 12 utterances / 170.720 seconds at their original 48 kHz sample rate.
All 12, plus three wrong-transcript controls, completed without errors, verified
endings or stop requests. Breeze alignment remains observational, so vocal
events and difficult speech are preserved. These recordings have no independent
word-time labels and do not demonstrate accurate timestamps.

Native unverified endings now remain **partial** through the controller and
Studio, even if a later fragment verifies. Text coverage alone cannot turn them
into a successful completion.

Evidence is retained under `D:/Temp/Codex/alignment-uncensored-20261009`:
`browser-controlled-03`, `browser-native-05-raw-validation.json`,
`heldout-engine-replay-01/report.json`, `engine-regression-02/report.json`,
`candidate-frozen-07.json`, `final-heldout-replay-01/report.json`,
`final-heldout-pcm-check-01.json` and `fantasy-stress-01/report.json`.
The browser harness also retains raw blocks and clock errors. The following
sections preserve the earlier 9 October baseline.

## Data and separation

The five development recordings contain two speakers: the
[LDC TIMIT public sample](https://catalog.ldc.upenn.edu/LDC93S1W), and LibriSpeech
`1272-128104-0000` through `0003` from
[hf-internal-testing/librispeech_asr_dummy](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy).
Durations are 2.925, 5.855, 4.815, 12.485 and 9.900 seconds.
Their local PCM16 WAV conversions were verified sample for sample.
Only TIMIT has independently labelled word boundaries:
[LDC93S1.wav](https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav) and
[LDC93S1.wrd](https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wrd).

A separate set was frozen before the policy change: **16 different speakers**,
eight from `test.clean` and eight from `test.other` of
[OpenSLR LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr).
Selection used eight evenly spaced row offsets per split, then the first
3–10-second mono recording from an unused speaker among the next 30 rows.
The development speaker was excluded. No model prediction influenced selection.
The manifest records row indices, transcripts, sources and WAV hashes.

The completion policy and implementation hashes were frozen before held-out
predictions. Both production source hashes remained unchanged throughout that
evaluation. No threshold was tuned against its failures.

## Models

English uses the documented
[Wav2Vec2 base 960h INT8 ONNX export](https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX),
revision `729c1a6730fb549c20a1c73a3d3f96f11020225e`, on two CPU threads.
Model SHA-256:
`1d9a366c27b2966625cd5035ac3db8847c53bba617169912bb251b42975a3a22`.

Multilingual matching uses
[Parakeet TDT 0.6B v3 INT8 ONNX](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx),
revision `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`, with
`onnx-asr==0.12.0` and two CPU threads. It retains native word/subword timing;
it does not provide independently measured character timestamps.

MMS uses the original
[TorchAudio MMS forced-alignment checkpoint](https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt),
SHA-256 `20ef12963ab4924bef49ac4fc7f58ad5da2ee43b2c11bc8c853c9b90ecdbc680`.
The acoustic model and tokenizer were preserved. Only the deprecated alignment
operation was replaced by the local vectorized CTC implementation.

## Development comparison

The previous implementation came from local commit
`35dc373167cefd7f67c8b90dc14bd6c6c3dbba2f`. Versions used identical recordings
and acoustic models, warmups, and alternating paired order. Every 160 ms input
prefix was processed before submitting the next prefix. These runs measure
causal stop decisions, not real-time coalescing or speaker latency.

Each recording was tested whole, truncated, against an unrelated transcript,
and with a repeated copy after 300 ms of silence. TIMIT was truncated at its
last word's labelled start; the other clips were cut at 60% of their duration.

| Result | Previous implementation | Final implementation |
| --- | ---: | ---: |
| Rejects missing suffix | 5/5 | 5/5 |
| Rejects unrelated transcript | 3/5 | 5/5 |
| Stops on whole recording | 5/5 | 5/5 |
| Stops before appended speech | 5/5 | 5/5 |

The earlier strict candidate policy rejected the valid 32-word development
sentence because a reduced internal “the” had very low CTC scores. The final
policy allows reduced short internal words while retaining overall support,
longer-word support, final-word support and competing-token margins.

Compared with the original unsafe rule, additional input needed for verification
was **0, 160, 160, 160 and 320 ms**. A confident word delimiter reduced the
earlier candidate policy's delay by 160 ms in two recordings. The remaining
verification delay is a measured safety tradeoff.

## Held-out results

There were **112 probes**, seven per recording. In addition to the four variants
above, these included a changed final word, deterministic additive Gaussian
noise at 10 dB signal-to-noise ratio, and silence. All used the frozen policy.

| Probe | Clean split | Other split | Total |
| --- | ---: | ---: | ---: |
| Whole speech: verified stop | 8/8 | 4/8 | 12/16 |
| Missing suffix: correctly rejected | 8/8 | 8/8 | 16/16 |
| Unrelated transcript: correctly rejected | 8/8 | 8/8 | 16/16 |
| Changed final word: correctly rejected | 8/8 | 8/8 | 16/16 |
| Silence: correctly rejected | 8/8 | 8/8 | 16/16 |
| Speech with added 10 dB noise: verified stop | 6/8 | 1/8 | 7/16 |
| Appended speech: eventually requests a stop | 8/8 | 6/8 | 14/16 |

There were **no observed false stops in the 64 negative controls** and no worker
errors. These correlated probes from 16 speakers do not establish a zero false
stop rate.

Eventual stopping is not the same as timely stopping. In appended-speech probes,
**12/16** detections arrived before the appended speech began. **13/16** proposed
cutoffs preceded it. Clip `3331-159605-0040` was detected at 7.04 seconds, after
extra speech started at 3.96 seconds, and proposed a late 7.25-second cutoff.
Clip `6070-86744-0007` was detected late but still proposed a cutoff before the
extra speech. Two cases never stopped.

The four whole-speech misses were `8280-266249-0029`,
`3331-159605-0040`, `3997-180297-0002` and `6070-86744-0007`.
Their audio remains available. The nine noisy-speech misses likewise retain
audio rather than authorizing an unsupported cutoff. The results support a
conservative guard, not a claim of universal completion detection.

## Real multilingual and MMS execution

German, Portuguese and Italian excerpts came from the
[official TorchAudio multilingual tutorial assets](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html):
`10349_8674_000087` (2.5 s), `6566_5323_000027` (4.6 s) and
`642_529_000025` (4.0 s). Their documented transcripts were retained; MMS used
the documented romanized Portuguese form. These are transcript references, not
independent human timing labels.

Real Parakeet inference completed the German excerpt. It reported partial
coverage on the Portuguese, Italian and English TIMIT recordings when recognition
differed from the supplied text, while preserving later matching words.
Examples include `imortal` versus `mortal` and `dark` versus `duck`.
Regression tests additionally reject substring matches such as `cat` inside
`caterpillar`, joined/split words, and suffixes crossing a window anchor.

MMS ran on all four recordings. On Torch/TorchAudio **2.6.0**, every character
frame span matched the legacy aligner. A separate environment with
Torch/TorchAudio **2.11.0+cpu** produced identical boundaries, with legacy
alignment functions replaced by raising stubs: **zero legacy API calls**.
This verifies the real checkpoint on both versions, not merely a mocked loader.

## Real TTS and queued playback

Seven requests were generated by the existing Qwen CPU service, using its
already registered Alex, Maya and Cleo voices. They covered English, German,
Italian, Portuguese and numeric text. The service reported
`Qwen3-TTS-12Hz-0.6B-Base`, Q8_0, CPU, native binding
`0.3.0+startup20260920.1`, mono 24 kHz PCM16. Its existing guard remained enabled;
the live service was not updated.

Captured PCM and its actual arrival times were replayed through this checkout's
`QwenEngine` at the native PCM input boundary. The final replay ran without the
other acoustic benchmarks. Aligned and unaligned natural runs alternated order
across cases, using identical captured input.

- **7/7** natural outputs verified completion and preserved their full duration.
- Their samples were identical to unaligned output before the final 25 ms fade.
- Paired first-queue timing differed by median **+0.246 ms**, range
  **−0.416 to +0.735 ms**. This small replay is not a hardware latency benchmark.
- With a controlled repeat appended after 300 ms, **6/7** runs verified completion.
  The Portuguese repeat case stayed partial and retained its audio.
- Even successful native stops delivered **20–660 ms** of surplus into the queue
  in the isolated replay. A completion event alone therefore does not prove
  that surplus was prevented.

Studio now applies verified, authorized trim metadata to still-scheduled audio,
with a short fade, exact fragment identity checks and no extra startup reserve.
Replaying these measured queue and event times through the actual Studio
scheduling functions and a deterministic AudioContext clock model removed all
queued surplus in the **six verified cases**. It correctly left the unverified
Portuguese case intact. Tests cover late arrival, stale sessions/requests,
following fragments and audio received before the context is ready.

This is real generated PCM through the current engine plus a tested player clock
model. It is not a deployment test, browser/device latency measurement or proof
that naturally hallucinated tails follow the controlled-repeat pattern.
Already played audio cannot be recalled. HTTP-only consumers do not receive
the WebSocket trim instruction.

## Boundary accuracy and computation

On TIMIT's 11 words (22 human-labelled boundaries), full-input alignment had
median absolute error **38.4 ms**, p95 **84.9 ms**, maximum **86.1 ms**.
The retained streaming ending did not cut before the labelled final word ended.
Only one sentence has independent word-time labels; the other recordings cannot
support a broad timing-accuracy claim.

The offline CTC dynamic program used identical 400-frame, 32-symbol emissions
and 100 targets, RNG seed 20261009, one warmup per version and five alternating
pairs. Median time fell from **623.9 to 13.8 ms**, with identical spans and scores.

For the 12.485-second real development recording, 25 adapter updates gave:

| Measurement | Full prefix | Bounded cursor |
| --- | ---: | ---: |
| Cumulative audio processed | 162.485 s | 87.525 s |
| Largest inference input | 12.485 s | 4.820 s |
| Total alignment worker time | 21.069 s | 11.168 s |
| Final words retained | 32 | 32 |

Boundary differences against the full-prefix method were median 0 ms, p95 20 ms
and maximum 40 ms. Those are method differences, not errors against gold labels.
These computation measurements are component results, not synthesis speedups.

## Reproduce

Use the local WAV/FLAC manifest and exact model files:

```console
python tools/benchmark_alignment.py --model /path/to/english-model --manifest /path/to/manifest.json --stress --output /path/to/new-results.json
```

For the paired development comparison, additionally pass
`--baseline-ref 35dc373167cefd7f67c8b90dc14bd6c6c3dbba2f`.
Only use a trusted local Git revision: its Python module is executed.

A manifest contains `{"cases": [{"id": "example", "audio": "/path/to/audio.wav",
"text": "known transcript", "source": "source URL"}]}`.
Optional `words` entries use `word`, `start_time` and `end_time`, and must come
from independent labels. The tool records hashes, policy, decisions and timings.
The additional timely-stop fields distinguish an eventual stop from a cutoff
before appended speech.

Real TTS capture requires an existing service and registered voices; edit
`tools/alignment_tts_requests.json` as needed. Supply its key through
`REALTIME_TTS_API_KEY`, never a checked-in file or command-line argument:

```console
python tools/benchmark_tts_alignment.py capture --base-url http://your-tts-server:18085 --manifest tools/alignment_tts_requests.json --output /path/to/new-capture
python tools/benchmark_tts_alignment.py replay --manifest /path/to/new-capture/manifest.json --model /path/to/english-model --multilingual-model /path/to/parakeet-model --output /path/to/new-replay
node tools/benchmark_studio_guard.mjs /path/to/new-replay/report.json /path/to/new-player-result.json
```

The tools refuse existing output targets. Keep model downloads, audio, manifests
and raw reports outside Git. This run's evidence directory contains
`development-completion-02.json`, `heldout-completion-01.json`,
`heldout-summary-01.json`, `completion-policy-frozen-01.json`,
`parakeet-real-01.json`, `mms-real-01.json`,
`mms-current-compatibility-01.json`, `tts-replay-isolated-02/report.json`,
`tts-isolated-summary-02.json` and `studio-guard-isolated-02.json`, together with
the provenance-bearing audio/model manifests.

## Regression checks and remaining limits

The required focused Python CI suite passed **564 tests**, with one optional
`qwentts_cpp` test skipped because that binding was absent locally.
All **34 Studio JavaScript tests** passed. The strict documentation build and
clean installation of the core and Breeze server/native candidate also passed.

Coverage includes partial final results, numeric projection at both timestamp
levels, CTC path equivalence, bounded windows, weak/moving completion evidence,
whole-word multilingual matching and the MMS API dependency. The Linux
production TTS service was left unchanged.

Confidence remains heuristic. Difficult speech, noise and recognizer revisions
can cause missed or late stops; the held-out counts above quantify that rather
than hiding it. Broad manually labelled timing accuracy, additional TTS engines,
acoustic loopback latency and naturally occurring hallucination tails are not
established by these checks.
