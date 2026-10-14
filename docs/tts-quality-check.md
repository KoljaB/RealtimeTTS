# Offline TTS quality checker

The rejected audio_anomaly detector and observer were removed. This checker
has no engine/playback integration, cloud requests, cancellation or trimming.

Run from the RealtimeTTS repository with its Python environment:

    .venv\Scripts\python.exe -B tools\tts_quality_check.py --wav PATH.wav --output result.json

For a hash-linked ASR result with genuine words/start/end, add --asr-json PATH.json.
Absent word spans are reported as word_timestamps_unavailable.

To produce a fresh single-clip ASR result with the existing Windows installation:

    C:\Python\Python312\python.exe -B -S tools\annotate_tts_faster_whisper.py --wav PATH.wav --output asr.json --language es

Then pass asr.json to the checker. The annotator uses only the verified cached
large-v2 weights, needs 5000 MiB free GPU memory, and changes no services.
It blocks an optional Torch import in this isolated process to avoid duplicate
OpenMP DLL loading; installed packages remain unchanged. Each invocation loads
and releases the model, so include model startup cost when timing this CLI.

Every contiguous quiet interval longer than 1 second is a long_silence warning
(default -50 dBFS, 20 ms frames). Leading and trailing silence count. A warning
can occur in acceptable speech, including the human-clean tutor clip.

The separate acoustic noise rule looks for weak resumed audio after a long gap:
0.5 s of audio, RMS at most 25% of preceding active-frame p95, at least 65% power
in 20--300 Hz, and spectral flatness at most .02 in 20--4000 Hz. Evidence values
and times are returned. ASR additionally reports audible intervals after the
last recognized word, with a .25 s margin and at least .20 s audible tail.

The benchmark runner accepts --language es/en/pt (default es). This sets BOTH
Qwen synthesis and faster-whisper transcription language; the corpus language
must match. English: --language en --corpus tools/data/english_tts_100.json.
The English corpus contains 90 one-to-three-word prompts and 10 longer sentences.
Voice and requested seed are selected independently with --voice and --seed.

This is an offline full-audio tool. evidence_available_s is an audio position,
not a measured live delivery timestamp. The acoustic rule can be computed from
the corresponding prefix; the ASR tail rule requires the complete ASR result.

The nine human-reviewed examples are development/regression evidence, including
previously inspected heldout files. They do not establish general accuracy.
no_anomaly_found means only that these rules found no evidence; never promote
it automatically to a human-confirmed clean label. This rule targets the observed
pause-then-weak-tonal-noise failure and does not cover every kind of gibberish,
noise without a pause, semantic error or mispronunciation. PCM16 WAV at >=8 kHz
is supported. Invalid/unsupported input raises an error rather than a clean result.

## Recorded checks on 2026-10-13

Both the acoustic rule and the ASR tail rule identify 3/3 human-confirmed
failures and produce zero noise flags on the four human-clean clips.
The clean tutor clip retains a separate 1.58 s silence warning.
All 111 retained WAVs were audited locally; the 100 unreviewed files are
not clean ground truth. Acoustic analysis wall time: median .652 ms,
p95 3.008 ms, worst 6.739 ms, excluding file loading, decoding and hashing.

Windows CUDA ASR used cached large-v2, faster-whisper 1.2.0, CTranslate2 4.6.0,
float16, one CPU thread and one worker. True final-word ends are .64 s
(spanish-wire-0, total20.48 s), .54 s (wire-2, total20.46 s), and 9.10 s
(original stream29, total19.305 s). Audible unrecognized tail totals are
6.76 s, 4.10 s and 1.10 s, respectively. No such tail flag occurs on the clean
clips. This does not treat silence alone as an acoustic hallucination.

Model loading cost 3.65 s. Once loaded, the two Spanish-word clips cost
.390/.391 s; other warm clips .829--2.398 s. The first clip cost 3.527 s
and includes first-use overhead. These are offline ASR timings, not live
streaming latency.

Raw evidence is retained in:
D:\Temp\User\tts-strange-noise-detector-20261013\checker-asr-gold.json
and its faster-whisper subdirectory. No model download, billing change,
new voice reference, live engine integration or server redeployment occurred.

## Isolated leading noise (2026-10-14)

The acoustic leading rule detects a short, weak low-tone island detached from
the following sustained audio. It requires ALL of: first island starts within
200 ms, duration at most 80 ms, quiet gap at least 200 ms, complete 120 ms
following active window, leading RMS at most 5% of the following window
(about 26 dB weaker), low-band fraction at least .65 and flatness at most .02.
The absolute audibility floor remains the configured silence threshold.

With word timestamps, a first recognized word ending by the resumed onset plus
one frame (20 ms by default) suppresses the rule conservatively: a quiet short
first word is not automatically extra noise. First-word
start alone is unusable for this boundary; cached Whisper results usually put
it at zero, including the confirmed noisy examples.

The event reason is isolated_weak_tonal_audio_before_speech. Its start/end
cover only the early island; a later ASR-tail event stays separate. Returned
evidence includes the quiet gap, both RMS levels, ratio and spectrum.
acoustic_evidence_available_s is the end of the required following window.
When an ASR end-time guard is used, evidence_available_s is conservatively the
full clip duration; this remains an offline checker, not measured live latency.

The two additional human-confirmed examples are Mas despacio (seed43, ref10-1)
and Tres nubes (seed42, ref10-1). Their opening sounds were labeled by the user
before the combined leading rule was frozen. This extends development evidence
to nine reviewed clips, not an independent heldout accuracy claim. Spectral or
envelope similarity alone does not label other recordings clean/noisy.
The rule is deliberately conservative: it considers only the first island and
the next sustained run. Intervening blips, overlapping noise, broad-band clicks,
and stronger leading sounds can still be missed. Mixed-channel artifacts and
inaccurate ASR word spans also limit this heuristic.

Source/ASR hashes and original user comments are retained in:
D:\Temp\User\tts-leading-noise-20261014\leading-human-manifest.json
