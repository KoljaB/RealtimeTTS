# Qwen prefix splice: A/B/C hearing experiment

Run from the repository on Windows using its Python environment:

```powershell
.\.venv\Scripts\python.exe tools\qwen_prefix_splice.py --api-key-file D:\path\to\tts-api-key
```

Alternatively set `QWEN_TTS_API_KEY` in the process environment. Credentials are
never saved in the experiment output. The default endpoint is the separate CPU
server at `http://192.168.178.22:18085`; capabilities must confirm CPU-only Qwen.
Use `--url`, `--voice`, and `--text` to override the defaults. German is the default;
for English pass both `--language english --phoneme-language en-us`.
An existing Miep TOML configuration can supply the credential via
`--config D:\Projekte\miep-codex\config.toml`; only `[tts].api_key` is read.

The demo sentence is “Heute Morgen gehen wir gemeinsam durch den Park und genießen
die frische Luft.” A receives `heute morgen` without punctuation; B receives the
whole sentence. Both requests use seed 42 and disable talker and subtalker sampling.
This does not guarantee identical pronunciation or prosody across different text
prefixes. The complete-request API still finishes native generation; omission of
punctuation is not an implementation of an open-ended native TTS input stream.

Phoneme alignment uses the CPU model
[`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft),
eSpeak phonemization and a CTC Viterbi path constrained to the known transcript.
It aligns real acoustic phoneme labels, not grapheme timestamps renamed as phones.
Dependencies: `numpy scipy torch transformers phonemizer` and an eSpeak NG shared
library visible to phonemizer. The first run downloads the alignment model into
the repository's `.cache/phoneme_alignment`. Phonemizer supports the
`PHONEMIZER_ESPEAK_LIBRARY` environment variable for an explicit library path.

The last phoneme of word one and first phoneme of word two bound a local search
in each recording. These CTC gaps are not assumed to be silence. A sliding RMS
window finds a low-energy plateau strictly within **±5 ms of each nominal midpoint**;
its position closest to that midpoint becomes the refined boundary. Lower minima
outside that narrow search are ignored. `--boundary-search-ms` can reduce the
correction (0 disables refinement), but cannot exceed 5 ms. Both offsets are saved
in the report. The default 10 ms crossfade straddles that boundary
in BOTH sources with complementary linear gains. Both boundary centers map to
the same output time, preserving the interval rather than compressing two cut
word edges together. `--crossfade-ms` changes the overlap. If it cannot fit the
search interval, the program fails explicitly instead of cutting into word anchors.
The CTC grid is still 20 ms and RMS refinement is a heuristic, not ground truth.
Listening remains necessary, especially when there is continuous voiced speech.

Outputs under `test_outputs/prefix_splice`:

- `A.wav`: the two words; `B.wav`: full sentence; `C.wav`: merged sentence.
- `A_boundary_left.wav`, `B_boundary_right.wav`: retained regions including overlap.
- `D.wav`: same centered crossfade, but no RMS refinement of alignment midpoints.
- `C_legacy.wav`, `report_legacy.json`: original faulty merge, preserved when upgrading.
- `A.alignment.json`, `B.alignment.json`: all phoneme and word boundaries.
- `synthesis.json`: source hashes, model, sampling and request durations.
- `report.json`: cuts, overlap, durations and limitations.

Press A/1, B/2, C/3 to play; another selection immediately replaces playback.
Space stops, Q/Escape exits. No Enter is needed. Reopen without inference:
Double-click `play_prefix_splice.cmd`, or run:

```powershell
.\.venv\Scripts\python.exe tools\qwen_prefix_splice.py --play-only
```

C/3 plays the RMS-refined merge, D/4 the merge without RMS correction; both use
the same 10 ms centered crossfade. E/5 plays the old merge when present.
Remix existing recordings and alignments without inference or network:

```powershell
.\.venv\Scripts\python.exe tools\qwen_prefix_splice.py --remix-existing --prepare-only --crossfade-ms 10
```

`--prepare-only` skips the keyboard player. `--align-existing` resumes after an
alignment dependency/download failure using hash-verified A/B WAVs and their
saved transcript. Use a fresh `--output` directory for a new synthesis run;
existing recordings are protected against accidental replacement.

This is an offline splice-quality MVP. Complete WAV request durations are not
TTFA measurements. Mutable playback buffers, LLM streaming and continuous prefix
regeneration are follow-up experiments.
