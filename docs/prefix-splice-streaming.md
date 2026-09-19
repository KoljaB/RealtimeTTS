# Experimental prefix-splice engine

`PrefixSpliceEngine` is an opt-in generator engine for `TextToAudioStream`. The
existing engines and server are unchanged. `QwenHttpSynthesizer` calls an existing
CPU Qwen server; an injected callable can supply another compatible synthesizer.
No model/backend substitution happens automatically.

```python
from RealtimeTTS import TextToAudioStream, PrefixSpliceEngine, QwenHttpSynthesizer
from RealtimeTTS.prefix_splice import PhoneAligner

backend = QwenHttpSynthesizer("http://192.168.178.22:18085", api_key,
                              voice="mira_v5_spark_de")
backend.prepare()  # authenticate and verify CPU model before playback
aligner = PhoneAligner("de", "/path/to/model/cache")
# Warm synthesis + alignment once before measuring latency.
audio = backend("Heute morgen")
aligner.align(audio, 24000, "Heute morgen")
engine = PrefixSpliceEngine(backend, aligner, commit_words=(1, 3))
stream = TextToAudioStream(engine)
stream.feed(llm_text_iterator).play()
```

Install the `prefix-splice` extra plus an eSpeak NG shared library usable by
phonemizer. The language-specific phone model is resident and word phonemizations
are cached. It runs on CPU. The tested Qwen model is 0.6B Base Q8_0 through
qwentts.cpp CPU, with fixed seed and greedy talker/subtalker sampling.

## What gets committed

| Text available | Synthesis | Playback queue | Retained privately |
| --- | --- | --- | --- |
| Two complete words | Exactly words 1–2 | Start through just before the first overlap | Overlap and remaining word 2 |
| Four complete words | Exactly words 1–4 | First overlap, then words 2–3 through the next guard | Second overlap and remaining word 4 |
| End of bounded block | Complete block | Second overlap and all remaining audio | Nothing |

Each overlap is 10 ms, centered on the corresponding word-boundary midpoint in
both source audios. Local RMS correction is independently restricted to ±5 ms.
CTC blank spans are **not** assumed to be silence. The complete overlap from the
earlier prefix remains private until the later prefix exists; queued PCM is
never edited, removed for replacement, or replayed.

Incomplete token fragments never trigger synthesis. A word is complete after a
delimiter or iterator EOF. Text should already contain spoken, spelled-out words;
numbers and tokens that require ambiguous normalization are rejected explicitly.
This MVP splits punctuation conservatively, not with a full abbreviation parser.

One-word input is synthesized at EOF. A two-word EOF flushes the retained original
tail without resynthesis. Three-word input performs one merge. Punctuation,
24 words or 240 characters close a block. A single token over the character limit
is rejected. After an early commit, 350 ms without another complete word flushes
that bounded block, preventing indefinite silence on an open but idle iterator.
Block boundaries are ordinary independent utterance boundaries; seamless prosody
between forced blocks is not guaranteed.

The schedule is configurable with `commit_words`, e.g. `(1,)` disables the second
speculative stage. There is one lookahead word per stage. There are at most three
syntheses per default block, rather than resynthesizing a growing unlimited sentence.
`max_block_words`, `max_block_chars`, `idle_flush_seconds`, `crossfade_ms`, and
`max_queue_seconds` configure the bounds. Input and committed-PCM queues apply
backpressure. Pause uses the normal player; stop cancels the local generation.

## Latency and failure contract

By default the initial two-word request still uses one complete WAV and one
alignment. Repeated partial alignments were slower on this short CPU request.
`stream_initial_prefix=True` (demo: `--stream-initial-prefix`) explicitly enables
partial alignment on that first request too. Neither setting promises a TTFA win.

**Following requests stream PCM while Qwen is generating it.** The HTTP client
validates mono 24 kHz s16le headers, reads available bytes with `read1`, and retains
split sample bytes. It does not wait for the complete sentence. When the next
boundary is accepted, it crossfades the private reserve and sends the available
suffix onward. During the final stage, subsequent PCM goes directly to playback.
`streaming=False` (demo: `--no-streaming`) selects the previous full-WAV path for
comparison. Injected backends need `stream(text)` and aligners `align_prefix(...)`
to use incremental commits; older callable backends keep their complete-audio path.

Partial alignment covers only the known words through the next word's first two
phonemes. A CTC free-suffix state absorbs later speech, avoiding the false assumption
that an unfinished buffer contains the entire sentence. A commit requires at least
120 ms of right context, anchor phone scores of at least 0.25, and **two identical
boundary snapshots with additional audio** (probes at least 160 ms apart in audio).
The 10 ms crossfade and ±5 ms RMS correction are unchanged. If partial evidence is
insufficient, the engine waits for more PCM or aligns the complete recording at EOF.
These confidence/stability checks are heuristics, not a phonetic ground-truth guarantee.

If the following synthesis is late, playback may underrun. The engine waits for the
correct suffix rather than replaying words, changing the requested text or inventing
audio. The `estimated_underrun` metric reports the deficit assuming continuous
playback from first queue commit. It is not an observed device underrun and is not
valid as an acoustic measurement during pause. Faster generation/alignment or an
explicit startup buffer is still needed to eliminate gaps; intermediate commits
can also underrun while the next held boundary is being established.

`metrics` separates synthesis wall time/RTF, alignment time, first queue commit,
and merge boundaries. The demo also records software playback callbacks; muted
playback drains faster than a real speaker. No number here measures physical
audible latency. Model warmup occurs before the timed demo.

Stop gates every queue insertion and discards the private overlap. The direct
Qwen HTTP adapter shuts down its active socket to interrupt response reads. Blocking HTTP,
alignment and user iterators run behind cancel-aware waits. A late operation can
finish in the background but cannot write PCM. Reuse is explicitly rejected until
the previous operation/input workers exit. HTTP operations have a 30-second timeout;
a user-owned iterator may remain blocked until its owner releases it. There is no
forced thread termination or hidden second request while the old request is active.

If alignment/synthesis fails before a handoff, a fully completed retained prefix
tail is emitted once and the exception is propagated. A reserve from an incomplete
stream is never recovered. After a streamed handoff, failures are terminal truncation:
already committed PCM cannot be retracted and the previous prefix must not replay.
`last_error` records the failure; remaining text is never claimed spoken. Insufficient
boundary space fails explicitly instead of overlapping the trimmed word edges.
HTTP EOF confirms transport completion, not a separate native success acknowledgement;
the current REST endpoint supplies no end-of-generation status frame. Disconnects,
truncated transfer framing, and odd final sample bytes are errors.

## Live demo

For a keyboard-triggered listening test on this Windows checkout, double-click
`start_prefix_splice_demo.cmd`. After server/model warmup, press **1**. The entire
text is fed immediately to `TextToAudioStream`; there are no artificial token
delays. The engine internally uses the 2-word / 4-word / full-block stages.
The printed TTFT is key acquisition to the first playback-start callback, immediately
before the first audio-device write. It excludes warmup and does not measure
physical speaker onset or remove device buffering. Space stops; Q/Escape exits.
Every press starts fresh synthesis. Remaining underruns are audible and the
estimated delivery deficits are printed after the run. `--text` changes the sentence;
`--stream-initial-prefix` also enables the optional initial partial-alignment path.

```powershell
.\.venv\Scripts\python.exe -X utf8 tools\prefix_splice_stream_demo.py --config D:\Projekte\miep-codex\config.toml --output test_outputs/my-prefix-run
```

The config supplies only the existing API key. The demo produces delayed text
tokens, plays through `TextToAudioStream`, and saves every source WAV, merged PCM
and timing report. Use `--muted` for an inference/integration check without speakers.
Use a fresh output directory each time. The saved `stream.wav` contains contiguous
received PCM; it does **not** insert silence corresponding to real-time stalls.

The earlier [A/B/C/D offline comparison](qwen-prefix-splice.md) remains available.

## Previous full-WAV baseline

The 2026-09-10 warmed CPU run used the default German sentence, 80 ms per incoming
word, and the existing 0.6B Base Q8_0 server. Three source requests contained
2, 4 and 13 words. An independent reconstruction from their WAVs and reported
crossfade positions matched every PCM16 sample of the 3.886125-second output.

- First queue commit: 1.545 s from simulated text start.
- First muted software playback callback: 1.601 s; no physical audio timing claimed.
- Alignment stages: 225, 291 and 518 ms.
- Estimated delivery deficits at the two handoffs: 1.46 and 2.69 s.

The implementation therefore proves staged synthesis/commit correctness, but this
backend configuration does not yet support gap-free immediate playback. The final
WAV omits those waiting gaps. Artifacts are in `test_outputs/prefix_stream_live_v3`.
The earlier per-word phonemizer reinitialization was removed; these are single-run
measurements, not a paired benchmark or a production latency guarantee.

## Verified incremental PCM run

The 2026-09-10 run in `test_outputs/prefix_pcm_live_v2` uses the same CPU model,
voice, text and sampling. All three source WAV hashes match the baseline above.
Both incremental handoff positions match the complete-alignment positions, and
every output PCM16 sample matches an independent reconstruction of the merges.

- Handoff after word one: 183 ms **before the second response finished**.
- Handoff after word three: 977 ms **before the final response finished**.
- First queue commit: 2.201 s; first playback callback: 2.259 s (muted).
- Estimated gaps: 1.09 s, 0.50 s and 1.31 s. The extra middle gap occurs while
  waiting for the next held boundary after the early first handoff.

This proves early following-audio delivery, not gap-free playback or improved
initial TTFA. An initial-stream experiment (`prefix_pcm_live_v1`) also committed
incrementally, but repeated alignment cost more on the short first request; this
is why it remains an explicit option. Transport timing varied between runs; these
numbers should not be treated as a paired performance benchmark.

## Continuous keyboard playback

The keyboard demo now defaults to `continuous_playback=True`. Immediate release
previously exhausted the playback queue inside words: the offline WAV did not
contain those delivery waits. Changing the crossfade cannot fix missing future
audio. The engine now retains the merged PCM privately until a bounded text block
is complete, then releases it through the existing bounded playback queue. The
2-word, 4-word and final syntheses and both crossfades are unchanged. Cancellation
discards the private PCM. A failed incomplete block is not played in this mode.
Buffering is capped at 60 seconds of PCM in addition to the existing text limits.

`first_buffer_commit` measures early private availability; `first_queue_commit`
and the keyboard TTFT describe the later playback release. `--immediate` selects
the previous experimental keyboard behavior with possible mid-word stalls.
This is a continuity fix, not a low-TTFA achievement. Long input can still pause
between bounded blocks, and device/scheduler failures are outside this guarantee.

The fresh CPU run in `test_outputs/prefix_continuous_live_v1` submitted the full
sentence without token delays, retained all three source requests, and reached
the first muted playback callback at 6.462 seconds. No physical speaker-onset
measurement or listening verification is claimed. The focused suite passed
39 tests, including byte-equivalent buffered versus immediate merges and stopping
while the second synthesis is blocked.
