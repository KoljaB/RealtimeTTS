# Reproducible Spanish TTS test: 50/100 cases

The default corpus contains 40 short texts with 1--3 whitespace-separated words
and 10 longer sentences. It includes ordinary phrases, questions, numbers, accents
and the known spelling triggers Hier-ba/Llu-via. Hyphenated forms count as one
word. The order is fixed; each TTS request uses seed 42 by default. The runner
also accepts the extended 100-case corpus (90 short, 10 longer) through --corpus;
the default remains the original 50-case fixture.

From the RealtimeTTS repository:

    C:\Python\Python312\python.exe -B -S tools\benchmark_spanish_tts_50.py --seed 42 --output-dir D:\Temp\User\spanish-tts50-next

For the extended fixed corpus, pass:

    C:\Python\Python312\python.exe -B -S tools\benchmark_spanish_tts_50.py --corpus tools\data\spanish_tts_100.json --seed 42 --output-dir D:\Temp\User\spanish-tts100-next

Use a new output directory for each complete run. --voice VOICE_ID selects an
already registered reference for a paired comparison. The model/backend is
verified as the configured LAN Qwen CPU service; all requests use language es,
max_new_tokens 256 and temperature .9 from the current Miep configuration.
These settings and server capabilities are saved, without authentication keys.

The script captures untrimmed PCM and producer receipts, performs the acoustic
check, then loads cached faster-whisper large-v2 CUDA float16 once for all
selected WAVs. Local ASR uses one CPU thread/worker. It makes no cloud/Gemini calls,
downloads, voice creation or service changes. The existing Windows Python
installation and at least 5000 MiB free GPU memory are required.

Outputs: corpus/run metadata, per-case WAVs, stream/chunk receipts, ASR word
times, checkpoint rows, summary.json, summary.csv and summary.md with audio links.
Noise suspicion, >1 s silence and ASR/synthesis errors remain separate.
No anomaly found is not a guarantee of clean speech.

--resume reuses matching recorded cases and never silently retries an errored
synthesis. Resume requires identical script/corpus/detector/settings/server
identity and checks each WAV hash and corpus entry. The report distinguishes
current-invocation wall time from stored per-case timing sums.

## Measured run on 2026-10-13

Model: Qwen/Qwen3-TTS-12Hz-0.6B-Base, Q8_0, qwentts.cpp CPU, 7 server threads.
Voice: scarlett_neutral_ed2bf8ef1054. Seed: 42.
Local ASR: large-v2, faster-whisper 1.2.0, CTranslate2 4.6.0, CUDA float16.

All 50 syntheses and all 50 ASR passes completed; no errors or unclassified cases.

| Group | Cases | Noise suspected | Silence only | No anomaly found |
|---|---:|---:|---:|---:|
| Short | 40 | 2 | 5 | 33 |
| Longer | 10 | 0 | 0 | 10 |
| Total | 50 | 2 (4%) | 5 | 43 |

The two flagged WAVs (Hier-ba and Llu-via) have exactly the SHA256 of the two
earlier user-confirmed failure recordings. Thus these two occurrences reproduce
known human-confirmed acoustic failures. Other clips have automatic evidence
only. The seed is requested through a supported parameter; the server does not
echo its applied seed. Exact WAV identity verifies repeatability for these two
cases, not a universal byte-identical guarantee across builds/references.

Five additional short clips have only >1 s silence: la lluvia, la musica,
Pan y sal, Casa azul, Ciento veinte.

| Timing | Seconds |
|---|---:|
| Whole Python process, externally timed | 124.088 |
| TTS including transport and recording | 89.982 |
| ASR model load | 3.644 |
| ASR inference over 50 clips | 24.968 |
| Acoustic detector over 50 clips | .0157 |

The remaining time includes startup, capability checks, model hashing and
report I/O. These are offline batch timings, not live streaming latency.
The source/fixture measurement hashes are retained in run.json/corpus.json.
Later corrections only made checkpoint reads explicitly UTF-8 and aligned a
descriptive corpus seed note; test texts and actual generation parameters
were unchanged and the measured run was not repeated.

Evidence:
D:\Temp\User\spanish-tts50-20261013\run-seed42-001\summary.md
D:\Temp\User\spanish-tts50-20261013\run-seed42-001\summary.csv
D:\Temp\User\spanish-tts50-20261013\run-seed42-001\external-runtime.json
