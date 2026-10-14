# Native compact language controls - 2026-10-14

RealtimeTTS TextToAudioStream now supports opt-in compact scopes such as
`<es>`, `<es slow>` (0.75x) and `<es fast>` (1.2x), with normal and numeric
alternatives and the long `<lang code="es">` form. The existing bounded parser,
shared scope stack and pitch-preserving tempo path are reused. Qwen native and
HTTP engines provide explicit language hooks. Optional language voice routes
restore their original voice, and caller-owned QwenVoice objects are preserved.

## Verified behavior

The selected local suite passed 132 tests, with one optional qwentts_cpp native
binding test skipped because that binding is absent from the Windows test venv.
After final player-state cleanup and correcting the expected-thread-exception
fixture, the 35 language/lifecycle checks passed again. Existing parser, tempo,
HTTP/native engine stubs and sentence-splitter compatibility were covered.

Tests exposed an error-path speed-state leak; cleanup now restores synthesis
and player speed as well as language and voice. Independent review added
deterministic late-feed, failed-producer and unsupported-fallback regressions.
Open scopes survive late feed in both synthesis paths, and failed workers exit
and expose errors. Native inline changes skip optional warmup synthesis; new
references can still require initial preparation. First failed selections and
JUnit results remain in the artifact directory; passing reruns do not erase
the initial failure. Routine staging/fixture/syntax mistakes were corrected
without changing existing unrelated sources.

## Live tests

The user first deferred Linux tests, then confirmed availability and explicitly
allowed short tests while another session used the host. The configured service
reported Qwen3-TTS 12Hz 0.6B Base Q8_0, qwentts.cpp ABI5, CPU backend at
192.168.178.22:18085. All four selected reference IDs were registered.

Two actual TextToAudioStream runs passed against that existing service:

- English explanation with Spanish slow/normal/fast phrase: 9.38054 s output.
- Portuguese explanation with German slow/normal/fast phrase: 9.67821 s output.

Each run made seven successful streamed HTTP synthesis requests with the
expected language/voice/speed sequence. Control text did not reach synthesis.
The WAV bytes matched audio callback bytes at mono PCM16/24kHz. Tempo durations
matched source duration divided by speed within the existing 120ms tolerance.
Both restored their original engine language and voice. First processed-audio
callbacks arrived after 303ms and 199ms; these are client API timing samples,
not acoustic onset or paired latency-regression measurements.

Two single-pass Qwen3.8-27B-Uncensored-Q4_K_M lesson prompts returned valid
compact controls, verified with the native parser. The Portuguese output
contained the unidiomatic greeting "Boa manhã"; it is retained as an open tutor
content failure in Miep's learning log. No response was regenerated. These two
responses do not establish broad model compliance or pronunciation quality.

## Artifacts and limits

`D:/Projekte/miep-codex/.runtime/compact-language-markup-20261014/` contains local
JUnit results, manifests, live_native.py, live-native-results.json, two native
WAVs and llm-syntax-live.json. The original source baselines are retained by
the guarded editor under D:/Temp/User/codex-edit-backups/edit-xjz31uo8 and
edit-oi6z5me0.

No Linux service, model, live voice selection or phone state was changed.
The tested path uses the new Windows client source and the existing Linux HTTP
service; no server restart was needed. Raw HTTP/WebSocket callers do not yet
parse these tags by themselves. Miep's own compact-tag/tempo integration is a
separate next step. No commit, publication or deployment was performed.
