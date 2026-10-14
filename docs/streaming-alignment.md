# CPU streaming timestamps and Qwen decoder guard

For the engine-independent Python API, playback modes and Voice Studio timeline,
see [Universal alignment](universal-alignment.md). This page covers model setup
and the Qwen decoder guard.

Install `pip install "realtimetts[alignment-cpu]"`. Download `model.int8.onnx`
and `tokens.txt` from
[the sherpa-onnx OmniASR 300M INT8 export](https://huggingface.co/csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12)
into one directory (about 365 MB).

Add `--alignment-model /path/to/model-directory` to the Qwen server.
The model is loaded and warmed at startup. It runs on two CPU threads in
one background worker. PCM production does not wait for recognition.
The guard retains a final 25 ms for fading after the initial PCM; it does not wait for recognition before delivering audio.

For Python, pass `alignment_model=path` and `on_alignment=callback` to
`QwenEngine` or `QwenCpuEngine`. Recognized words also enter the existing
`engine.timings` queue used by TextToAudioStream's word callbacks.
Character/word results can arrive after their audio has played.

The WebSocket speech stream emits these JSON events beside the PCM:

- `alignment`: new `words` and `characters`, each with `start_time`,
  `end_time` (seconds), and acoustic `score`; characters also have `index`.
- `alignment_complete`: the supplied text has full timestamp coverage and
  passes the configured `CompletionPolicy`. CTC uses word/suffix support and
  target-versus-competing-token log margins from the same emissions. Parakeet
  uses independently recognized whole words and their scores. A confident CTC
  word delimiter can confirm the first supported endpoint. Otherwise endpoints
  need two stable observations or at least the existing 250 ms ending allowance
  as context. Natural end-of-input can finalize a supported candidate without
  another inference. No amplitude-based quiet tail is required.
  `end_time` is the last character/word boundary; `trim_end_time` adds at least
  250 ms. Only `stop_requested: true` authorizes a downstream player to trim.
- `alignment_stalled`: optional no-progress cutoff, only when explicitly enabled
  with `stall_seconds`. This does **not** mean the sentence was completed.
- `alignment_done`: coverage completion, `completion_verified`, finish reason,
  acoustic margin (when available), and inference measurements. A fully aligned
  but insufficiently supported result finishes as `text_aligned_unverified` and
  never requests a stop.
- `alignment_error`: timestamps/recognition became unavailable; PCM continues.

Server events contain `session_id`, `request_id`, and one-based
`fragment_index`. Ordered `fragment_start`/`fragment_end` events delimit each
fragment's binary PCM, including empty fragments. Their `sample_rate_hz` and
unique request IDs let downstream players trim only that fragment, even when
alignment arrives after a successor. Server-added gap silence is outside these boundaries. Timestamps are relative to that fragment's PCM, before
server-added inter-fragment silence. They are not speaker playback timestamps.
Character indices refer to NFKC/lowercased text with punctuation converted to
spaces and repeated whitespace collapsed. HTTP PCM/WAV responses do not
carry JSON metadata.

With the guard enabled, acoustically supported, stable completion or an explicitly enabled stall
requests a native decode stop, preserves the 250 ms ending allowance, and
fades the retained final 25 ms. Completion follows the configured aligner and
does not classify amplitude as noise. It does not inject an EOS token or
automatically replay words already heard.
A stall remains an explicit event so the caller can choose recovery.
Recognition lag and the native callback interval add to detection time; audio
already played cannot be recalled. An enabled stall threshold is not an audible upper bound.

Studio also honors `completion_verified: true` plus `stop_requested: true`
for WebSocket playback, even when timestamp highlighting is off. It fades and
stops audio still scheduled beyond `trim_end_time`, using the exact session,
request and fragment identities. Future chunks from that fragment receive the
same cutoff; stale events cannot trim a successor. This adds no startup reserve.
Existing successor scheduling and inter-fragment gaps are preserved. The WAV
download retains the original received PCM. HTTP responses have no trim metadata,
and an unverified ending never authorizes this player cutoff.

Use `--no-hallucination-guard` (Python: `hallucination_guard=False`) for
timestamps without automatic stopping. A custom `StreamingTextAligner` can
set the analysis interval or explicitly enable a stall timeout. Stall stopping
is disabled by default: a recognizer mismatch must not abort valid speech.

CTC confidence is evidence, not a guarantee of intelligibility. The model can
miss names, numbers, short phrases, or unfamiliar pronunciation. The English CTC path aligns the known synthesis text directly; a separately
decoded spelling does not veto its endpoint. Individual low-probability letters
are allowed within the policy's limits because valid pronunciations can differ
from the model's spelling. The final word needs stronger support. These are
explicit heuristic thresholds, not calibrated probabilities of correct speech.

Defaults require at least 80% of alphanumeric characters at probability 0.35,
at least 50% support within each word of four or more characters, and 75%
support in the final word. Short internal words can be reduced in fluent speech;
they still count toward overall support. Both available CTC mean log margins
must be nonnegative. The delimiter fast path requires probability 0.9 before
any further decoded letter, within the model's right-context interval.

Override them with `StreamingTextAligner(completion_policy=CompletionPolicy(...))`
only after evaluating representative voices and failure cases. Spaces are word
boundaries, not spoken characters. Alignment is an estimate, not a proof against misalignment.
Assess it with the intended voices. Parakeet still matches recognized words.

## Faster English model

For English, the INT8
[Wav2Vec2 base 960h ONNX export](https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX)
can recognize short phrases earlier than OmniASR. Put its
`onnx/model_quantized.onnx` (as `model_quantized.onnx`), `vocab.json`, and
`config.json` in one directory, then use
`--alignment-model /path/to/english-model --alignment-language english`.
Other languages continue synthesis without this English detector.

Analysis runs at fixed 160 ms audio boundaries. Reaching the supplied text's
end creates a candidate; acoustic support and endpoint stability authorize a
stop with a 250 ms ending allowance and a 25 ms fade. It does not require a second transcript or
a quiet tail. A pause with unfinished text does not authorize a stop.
No input punctuation is changed.

## Fast multilingual CPU matching

Install `realtimetts[alignment-multilingual]` and pass
`--alignment-multilingual-model /path/to/parakeet-v3-int8`. The model directory
contains `encoder-model.int8.onnx`, `decoder_joint-model.int8.onnx`,
`vocab.txt`, `config.json` and the exported NeMo frontend `nemo128.onnx`.
The decoder is onnx-asr 0.12, with two non-spinning CPU threads. No GPU or LLM
is used. Model loading/warmup happens at service startup.

Supported guard languages are English, German, Spanish, French, Italian,
Portuguese and Russian. Add the existing `--alignment-model` and
`--alignment-language english` options to retain the faster English model;
the multilingual model serves the other six languages. Both share one worker,
so separate language models never run concurrently.

Parakeet emits **word timestamps**, based on native subword times at 80 ms
resolution. It does not advertise independently measured character timestamps.
`features.alignment_language_timestamps` reports the levels for each language.
Matching preserves word boundaries: `cat` does not match `caterpillar`, and
`a b` does not match `ab`. A recognition mismatch can leave a gap while later
exact matching words still receive timestamps. Missing text leaves synthesis running. Recognized surplus
after the supplied text no longer hides its completed endpoint. The same
coverage/support/stability contract applies across languages, although the
backend evidence differs. This is not a guarantee against all hallucinations.

The first PCM chunk is published before per-utterance model inference starts.
The server releases analysis after the first WebSocket send or HTTP generator
yield; standalone QwenEngine releases it after its first PCM queue publication.
PCM publication never waits for a recognition result. Multilingual checks run
beside generation at a 320 ms audio cadence, coalescing if inference falls behind.

## Recorded validation

See [the recorded evaluation](alignment-validation.md) for separate development
and held-out speech, negative/noise controls, real MMS/Parakeet execution,
measured TTS queue behavior, computation results and remaining limits.
These checks do not establish calibrated confidence or universal accuracy.
