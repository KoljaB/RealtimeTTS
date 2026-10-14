# Alignment for every engine

Alignment lives on `TextToAudioStream`, so the same API works with every
`BaseEngine`: local or cloud, sentence or generator input, PCM or MP3 output.
It uses native timestamps when the engine supplies the requested level and a
separately configured forced aligner otherwise. Alignment is optional.

## Start with the shared API

```python
from RealtimeTTS import AlignmentConfig, TextToAudioStream

# engine is any configured RealtimeTTS engine.
config = AlignmentConfig(
    source="auto",
    mode="live",
    level="word",
    model_dir="models/alignment",  # optional when native timings are available
    language="english",
)

with TextToAudioStream(
    engine,
    alignment=config,
    on_word=lambda word: print(word.text, word.start_time, word.text_start),
    on_alignment=lambda snapshot: print(snapshot.status, snapshot.revision),
) as stream:
    stream.feed("Hello from any engine.").play()
    result = stream.alignment.wait(timeout=30)
    result.export("speech.json")
    result.export("speech.srt")
    result.export("speech.vtt")
```

`alignment="auto"` is the shorthand. With native timings it needs no alignment
model. Otherwise, supply `model_dir` or `aligner`; without either, the existing
optional omniASR Python backend is created lazily. Missing dependencies or
unavailable timestamps are reported on the timeline while speech continues.

For the CPU ONNX backend, install `realtimetts[alignment-cpu]` and place the
model files described in [CPU model setup](streaming-alignment.md) in
`model_dir`. Model languages and timestamp levels still apply: a word-only
model does not acquire character accuracy from this API. The library does not
estimate character boundaries by dividing a word into equal pieces.

For an existing model object, use `AlignmentConfig(aligner=my_aligner)`.
The custom protocol is `align_words(text, mono_float32_audio, sample_rate)`;
character mode requires `align_characters(...)`. Existing `StreamingTextAligner`
instances can be reused. Models are optional and are not bundled in releases.

MP3 engines additionally require `pip install "realtimetts[alignment-audio]"`.
Compressed packets are decoded incrementally once; alignment and playback then
consume that same decoded PCM. Normal playback without alignment keeps its
existing audio path.

## Choose the latency policy

| Mode | Behavior |
| --- | --- |
| `live` | Speech does not wait for alignment. Results can arrive after the corresponding audio. |
| `synced` | Playback may wait once at the beginning of each fragment, up to `max_buffer_seconds` (default 0.25). It continues and reports `behind` when the budget expires. |
| `export` | Collects final alignment after synthesis; `play()` waits for it before returning. |

Sync mode improves the chance of timely highlighting; it cannot guarantee
timestamps will beat a slow backend. A full-utterance custom aligner produces
results after that utterance is synthesized. CPU ONNX adapters can publish
provisional results while PCM arrives. `prepare()` on `stream.alignment`
constructs the selected fallback model ahead of synthesis; model-specific
warmup remains the backend's responsibility.

CPU adapters retain a supported transcript prefix and re-analyze an overlapping
window (about 3.2 seconds of context, capped at 12 seconds per inference).
Committed PCM is discarded from the analysis buffer. If no trustworthy anchor
can advance within that window, alignment becomes unavailable and speech
continues. Full-utterance custom aligners keep their existing protocol.

The analysis queue and per-fragment audio storage are bounded. Defaults are
eight pending fragments and 120 seconds per fragment. When either limit is
exceeded, that fragment is marked unavailable and speech continues. Configure
`max_pending`, `max_audio_seconds` and `update_interval` on `AlignmentConfig`
for your workload.

## Read one timeline

`stream.alignment` exposes:

- `snapshot()`: an immutable `AlignmentSnapshot`.
- `words`, `characters`, `current_word`, `status`.
- `at(seconds, level="word")`: the span covering an audio position.
- `wait(timeout=...)`: wait for final metadata; raises `TimeoutError` if it is
  still pending.
- `export(path)`: write JSON, SRT or WebVTT, inferred from the extension.

Each `AlignedSpan` contains `text`, `start_time`, `end_time`,
`text_start`, `text_end`, `source`, `state`, `level`, optional `score`,
and `fragment_id`. `word` and `character` are convenience aliases for text.

Audio times are seconds from the start of the complete output, including
inserted pauses and engine fallback. Text ranges are half-open Unicode
codepoint offsets into the exact input. JavaScript clients must use
`Array.from(text)`, not UTF-16 string indices. A mapping that cannot be
established is `null`; the library does not fabricate exact offsets for an
unknown rewrite.

`source` distinguishes native provider timings from model-derived timings.
Model results may be `provisional` until the fragment is complete. A snapshot
has `final`, `revision`, `duration`, `coverage` and `errors`. Coverage is
the fraction of normalized spoken symbols covered at the requested timestamp
level; punctuation and whitespace are excluded. A final snapshot can be
`ready`, `partial`, `unavailable` or `cancelled`. `final` means processing
has ended. `ready` requires full coverage; a nonempty prefix is `partial`,
and its available timestamps remain exportable. Coverage measures transcript
mapping, not acoustic correctness or permission to stop speech. SRT/VTT group the available word spans into readable cues; JSON
preserves all metadata, including characters and errors.

`on_alignment` receives coalesced snapshots on a background notification
thread. `on_word` and `on_character_spoken` follow the output playback clock.
Callbacks should stay quick and hand UI work to the UI's own thread. Muted
output advances a logical output clock. Metadata that arrives late during
playback is delivered when available, with its original timestamps; after
playback it remains available through the timeline. For highlighting, query
`current_word` or `at()` rather than treating a delayed callback as current speech.

Pause freezes the playback position. Stop cancels the active alignment
generation; a new `play()` resets the timeline. Old worker results cannot enter
the new generation. Use the context manager or call `stream.close()` to release
the optional alignment workers and restore the engine queues. The caller still
owns the engine itself.

The existing `align_words`, `word_aligner` and `alignment_blocking` arguments
remain accepted. New applications should use `AlignmentConfig`;
`alignment_blocking=True` maps to bounded sync mode.

## Voice Studio

Studio has alignment source, word/character and latency controls alongside
the existing engine controls. Its timeline supports live highlighting,
a waveform, word selection, source/timing inspection and JSON/SRT/VTT downloads.
Recorded audio can be replayed from a selected word. Pauses and replay use the
same browser audio clock as highlighting. In synced mode Studio keeps the
configured playback lead (0.25 seconds by default); an explicit zero disables
that extra lead. The WebSocket servers send PCM promptly, and the client owns
its playback policy. Pause always stops browser playback; engines without native
pause support can continue generating audio in the background.

Alignment uses the WebSocket speech endpoint. Select a backend and language
that supports the requested level; an unavailable backend is shown explicitly.
Qwen and Breeze servers reuse their configured CPU models and retain their
existing engine-specific controls.

To expose the same Studio for another engine, use the optional generic host:

```python
import uvicorn
from RealtimeTTS.server import create_app
from RealtimeTTS.alignment import OnnxCTCBackend, StreamingTextAligner

aligner = StreamingTextAligner(
    OnnxCTCBackend("models/alignment", language="english")
)
app = create_app(engine, aligner=aligner)
uvicorn.run(app, host="127.0.0.1", port=8000)
```

Install `realtimetts[studio]` and your chosen engine/alignment extras first,
then open `http://127.0.0.1:8000/studio/`. The generic host receives an already
configured engine; it does not load every provider or require their credentials.
Use `api_key=` when exposing it beyond a trusted local machine.

## WebSocket metadata

The shared API accepts an `alignment` object on the initial configuration:

```json
{
  "type": "config",
  "voice": "your-voice",
  "response_format": "pcm",
  "alignment": {
    "source": "auto",
    "mode": "live",
    "level": "word",
    "max_buffer_seconds": 0.25
  }
}
```

Requests may select those four fields, not server model paths or Python objects.
Configure models on the server. HTTP audio-only requests cannot carry the
alignment metadata contract and reject an enabled alignment request.

Shared `alignment` events are **replacement snapshots**, identified by
`revision`, not append-only token deltas. The event carries the fragment ID,
Unicode codepoint offsets and `time_base="fragment"`. Combine it with the
fragment's playback start to place spans on the browser timeline. An
`alignment_done` event reports completion or failure. Existing engine-specific
alignment streams without a revision remain legacy deltas.
