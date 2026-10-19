# Language routing

Use one `LanguageRoutingEngine` when different languages need different providers.
The router selects a provider for each language span, while `TextToAudioStream`
keeps one playback queue.

## Python API

```python
from RealtimeTTS import LanguageRoute, LanguageRoutingEngine, TextToAudioStream

# breeze and qwen are already configured engines.
router = LanguageRoutingEngine(
    default_language="en",
    routes={
        "en": [breeze, qwen],
        "zh": [breeze, qwen],
        "*": [qwen],
    },
    pre_first_pcm_timeout=30,
)
stream = TextToAudioStream(router, language_tags=True, speed_tags=True)
stream.feed(
    "That tree is <es>el árbol</es>. Try saying it. "
    "Listen slowly: <es slow>Vamos al parque.</es>"
).play(
    comma_silence_duration=0.15,
    sentence_silence_duration=0.30,
    language_transition_silence_duration=0.08,
)
```

Route order is fallback order. An exact language route takes precedence over
`"*"`; the wildcard is used only when there is no exact route. ISO codes such as
`en`, `es`, `de`, and their full language names are accepted. The supported
languages are English, Chinese, Japanese, Korean, German, French, Spanish,
Italian, Portuguese, and Russian. Explicit tags override detection.

Providers must expose **24 kHz, mono, signed 16-bit PCM** and implement language
selection. The router rejects incompatible formats; it does not resample them.
Providers belong to the caller and must not be used concurrently outside the
router. Shutting down the router does not shut down those providers.

Fallback happens only before any PCM has been committed. Failure after audio
starts ends that synthesis rather than replaying the sentence. A timeout stops
the provider; a worker that fails to stop is quarantined until it exits.
Cancellation never triggers fallback. The timeout is configurable because cold
model startup can take much longer than warm synthesis.

### Provider-specific voices and delivery

A logical voice can map to different registered voices on different providers:

```python
b = LanguageRoute(breeze, voice=breeze_voice, voices={"tutor": breeze_voice})
q = LanguageRoute(qwen, voice=qwen_voice, voices={"tutor": qwen_voice})
router = LanguageRoutingEngine({"en": [b, q], "*": [q]})
router.set_voice("tutor")
router.set_instructions("warm, encouraging")
```

Instructions go only to providers advertising direct voice control. Qwen's
reference voice does not receive Breeze delivery instructions. Reset manual
Python instructions with `router.set_instructions("")` when starting a new
reply. Freeform bracket cues such as `[warm, encouraging]` are parsed by
Miep. Python streams recognize only explicitly registered instruction tags,
as described below; unregistered cues remain literal text.

Provider sampling settings remain on each provider. A `RemoteTTSEngine` route
can receive `seed=42` and `max_new_tokens=256` at construction; these values are
sent only in its initial WebSocket config. The router does not apply one
provider's temperature, token limit, or seed to another provider.

## One server endpoint for Miep

Install `RealtimeTTS[routing]` on the routing host. Keep the existing Breeze
and Qwen services on their own hosts, models, backends, and sampling settings.
Both provider services must run a RealtimeTTS version advertising
`per_request_silence` and `segmented_narration`. The routing adapter checks
these at startup.

Example `routing.json`:

```json
{
  "default_language": "en",
  "pre_first_pcm_timeout": 30,
  "providers": {
    "breeze": {
      "url": "http://127.0.0.1:18084",
      "voice": "your-breeze-voice",
      "voices": {"tutor": "your-breeze-voice"}
    },
    "qwen": {
      "url": "http://192.168.178.22:18085",
      "voice": "your-qwen-voice",
      "voices": {"tutor": "your-qwen-voice"},
      "api_key_env": "QWEN_API_KEY"
    }
  },
  "routes": {
    "en": ["breeze", "qwen"],
    "zh": ["breeze", "qwen"],
    "*": ["qwen"]
  },
  "silence": {"comma": 0.15, "sentence": 0.30, "transition": 0.08}
}
```

Use `api_key_env` or `api_key_file` for each provider's existing credential.
Protect the routing endpoint separately with `REALTIMETTS_API_KEY`.
Non-loopback binding requires that key.

```sh
realtimetts-language-server --config routing.json --host 127.0.0.1 --port 18086
```

Set Miep's TTS endpoint to this router, select logical voice `tutor`, and map
its language voices to configured logical names. Keep `tts.engine = "qwen"`;
routing is discovered through capabilities. Set
`natural.voice_control_mode = "direct"` and `natural.emotions_enabled = true`
for freeform English delivery cues, and keep the existing language tutor setting.
Delivery cues require the endpoint's `direct_voice_control` capability. Miep no
longer sends shared sampling overrides when the server advertises language routing.

Miep can then stream this without extra model calls or buffering the whole reply:

```text
[warm, encouraging] That tree is <es>el árbol</es>. Try saying it.
Listen slowly: <es slow>Vamos al parque.</es>
Now normally: <es>Vamos al parque.</es>
A little faster: <es fast>Vamos al parque esta tarde.</es>
[pleased] Nicely done!
```

The delivery cue lasts until another cue or the end of the current assistant
reply. Sentences and foreign-language spans do not reset it. A new reply starts
with no previous cue. Language tags scope language and tempo until their closing
tag. Cues are metadata and are not spoken.

For an English voice-design route, set the Breeze provider's voice to
`__breeze_voice_design__` and set its `instructions` to the fixed voice
description. Delivery cues are appended to that description. Register reference
voices on their provider services before configuring logical mappings; the
routing server's voice catalog is configuration-based and does not proxy voice
registration or Miep's dynamic voice-library management.

Explicit language spans do not need automatic detection. For untagged `Auto`
fragments, install `fasttext-predict` and set `language_detector_model` in the
routing configuration to an existing FastText model path. Without a detector,
untagged `Auto` text uses `default_language`.

Miep's walking tutor (`natural.language_tutor = "es"` or `"pt"`) requires
this detector even when phrases use explicit tags. Keep
`natural.other_language_response = "auto"` and TTS `language = "Auto"`;
the endpoint must advertise `mixed_language_fallback` and
`stream_controls.segmented_narration`. Detection supplies fallback for
untagged fragments; explicit language spans remain authoritative.

## Pauses at language boundaries

`<es>el árbol</es>. Try saying it.` assigns the outer period to the preceding
Spanish speech. A comma outside the tag similarly uses the comma pause. A bare
language switch adds the transition pause, including when returning to English.

The pause is the largest applicable duration, not their sum. An already emitted
comma pause is only topped up if a later period requires more silence. There is
no transition pause before the first spoken fragment. Set `transition` to
zero to disable that part of the policy.

The router owns these explicit pauses. Remote providers receive per-stream
zero pause settings, so their server pauses are not added a second time.
Miep trims only the beginning of the logical reply, preserving later gaps across
language and tempo changes. This fixes missing explicit gaps; it does not by
itself establish that a native model never clips a final phoneme.

## Delivery instructions within a reply

Register named delivery controls on the Python stream, then place them between
sentences:

```python
stream.add_instructions({
    "normal": "Speak at a brisk conversational pace.",
    "surprised": "Sound surprised and delighted.",
    "whispered": "Whisper softly but clearly.",
})
stream.feed(
    "[normal]Welcome back! [surprised]Oh wow, you learned that already! "
    "[whispered]Here is a little secret. [normal]Let us continue."
).play()
```

Each registered tag ends the preceding synthesis fragment and changes the
instructions for subsequent fragments, including across language scopes.
Tags are removed before synthesis and alignment. Unknown tags remain literal;
registered names must not collide with voice, pause, or built-in controls.
These are point controls, so use another registered tag to change delivery.
Completion, cancellation, or failure restores each engine's prior instructions.

The engine must support `set_instructions` and either `get_instructions()` or a
public `instructions` attribute. Unsupported engines raise an error before the
controlled text is synthesized. `LanguageRoutingEngine` forwards instructions
only to routes that support them; a plain Qwen CPU route keeps its reference
voice without receiving unsupported delivery prompts. Model delivery is still
probabilistic. Use explicit tempo controls when a consistent speed is needed.

## Vocal cues and alignment

Register one-shot sounds separately from sticky delivery instructions:

```python
stream.add_instruction("whispered", "Whisper very quietly.")
stream.add_vocalizations({
    "sigh": "Exhale a gentle sigh.",
    "laugh": "Brief laughter.",
    "clear_throat": "Clear your throat once, briefly.",
})
stream.add_pause("pause", 0.7)
stream.feed(
    "[whispered]A little secret. [laugh]Still whispered."
    "[pause:0.7][sigh][pause:0.7]"
).play()
```

Both bulk methods accept mappings, merge with existing registrations and return
the stream for chaining. A batch is validated before any registration changes;
invalid names, conflicting tags or invalid directions leave the prior mapping
intact. Use `None` for a vocalization with no custom direction. The singular
`add_instruction()` and `add_vocalization()` methods remain available.

Each registered sound is synthesized once, at its position in the stream.
Its optional instruction applies only to the sound and restores the preceding
delivery even on failure or cancellation. Without an instruction, the sound
inherits the current delivery. Language, voice and speed are preserved.
No extra automatic punctuation pause is added after a sound. The incremental
parser accepts tags split across input chunks and flushes preceding speech.

Breeze and the remote adapter translate supported names to native vocal cues;
spaces in provider names become underscores, such as `clear_throat`.
The remote adapter uses the provider's advertised `vocal_tags`. Unsupported
sounds or directions raise an error before sound synthesis. Every fallback
provider on the selected language route must support the same cue. A registered
sound is never sent as ordinary prose to a route that cannot perform it.

Sound tags are excluded from spoken alignment text, while their audio duration
still advances the timeline. `TerminalHighlighter(input_text=...)` can display
the original tags without shifting subsequent word offsets. Unregistered tags
retain the parser's existing literal-text behavior.

For a remote provider advertising `vocal_tags` and UTF-16 non-speech ranges,
`RemoteTTSEngine` preserves those exclusions for local alignment. The Python
language router forwards an exclusion only when all fallback providers on the
selected route agree, so a fallback that speaks the text cannot silently lose
its timestamps.

With Breeze, `(chuckle)` is a vocal cue, not a word to force-align. To give a
standalone chuckle a separate audio interval in a Python stream with
`language_tags=True`, use `Welcome! <en>(chuckle)</en> Let's continue.`. The cue
remains in the source text but has no word or character timestamps. Speech
alignment resumes on the next fragment. This marks an explicit synthesis cue;
it does not locate arbitrary laughter inside a mixed speech fragment.

## Streaming behavior

Connections to each remote provider are reused across that reply's fragments.
Pause, resume, and cancellation are forwarded to the active provider. Deferred
alignment callbacks stay attached to their original fragment and stream.

Native word alignment is advertised only when every configured provider can
supply it. Otherwise this CLI configuration does not provide automatic word
alignment. A custom Python server can configure a universal alignment backend
separately. Neither route selection nor silence insertion waits for alignment
before forwarding PCM.
