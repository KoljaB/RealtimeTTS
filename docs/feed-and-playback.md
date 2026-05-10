# Feed And Playback

The main RealtimeTTS workflow is:

```python
stream.feed(text_or_iterator).play()
```

`feed()` queues text. `play()` or `play_async()` synthesizes and plays it.

## Feed Text

Use a string when all text is already available:

```python
stream.feed("Hello. This text is ready now.")
```

Use an iterator when text arrives over time:

```python
def chunks():
    yield "The first part arrives. "
    yield "The second part follows."


stream.feed(chunks())
```

Each `feed()` call returns the stream, so chaining works:

```python
stream.feed("Hello from RealtimeTTS.").play()
```

## Play

`play()` blocks until synthesis and playback finish:

```python
stream.feed("Speak this now.")
stream.play()
```

Useful parameters include:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `fast_sentence_fragment` | `True` | Allows early synthesis before a full sentence is complete. |
| `buffer_threshold_seconds` | `0.0` | Waits for enough buffered audio before requesting more text. |
| `minimum_sentence_length` | `10` | Joins very short fragments before synthesis. |
| `minimum_first_fragment_length` | `10` | Minimum length for the first yielded fragment. |
| `output_wavfile` | `None` | Writes synthesized audio to a WAV file. |
| `on_audio_chunk` | `None` | Receives raw audio chunks as they are ready. |
| `tokenizer` | `"nltk"` | Sentence tokenizer for streamed text. |
| `language` | `"en"` | Sentence-splitting language. |
| `muted` | `False` | Disables local speaker playback for this call. |
| `force_first_fragment_after_words` | `30` | Forces the first fragment after this many words. |

## Play Async

`play_async()` starts playback in a background thread:

```python
import time

stream.feed("Background playback.")
stream.play_async()

while stream.is_playing():
    time.sleep(0.1)
```

`play_async()` has almost the same parameters as `play()`, but currently defaults
`fast_sentence_fragment_allsentences` to `True`; `play()` defaults it to
`False`. Keep this difference in mind when comparing latency between sync and
async examples.

## Pause, Resume, And Stop

```python
stream.pause()
stream.resume()
stream.stop()
```

Pause and resume control playback. Stop asks the active engine to stop, aborts
current playback, joins the async thread if needed, and resets the internal text
iterator.

Pause and resume may not behave the same with engines that use the mpv playback
path for compressed audio.

## Text And State

```python
if stream.is_playing():
    print("Still speaking")

print(stream.text())
```

`is_playing()` reports whether synthesis or playback is still active. `text()`
returns the accumulated generated text.

## Inline Voice And Pause Tags

The stream can register voice-switch and pause tags. The exact voice object or
identifier depends on the active engine.

```python
stream.add_voice("narrator", voice)
stream.add_pause("short-pause", 0.25)
stream.feed("[narrator]Hello.[short-pause]Now continue.")
stream.play()
```

The default delimiters are `[` and `]`. Use `set_voice_tag_delimiters()` if your
input format needs different markers.
