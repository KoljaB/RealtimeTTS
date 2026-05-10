# LLM Streaming

RealtimeTTS works well with LLMs because `feed()` accepts iterators. You do not
need to wait for a full assistant response before audio can start.

## Provider-Neutral Pattern

Wrap your LLM client's streaming response so it yields only text chunks:

```python
from RealtimeTTS import TextToAudioStream, SystemEngine


def llm_text_chunks(prompt):
    # Replace this body with your LLM client's streaming iterator.
    yield "Here is the first streamed phrase. "
    yield "The next phrase can arrive while audio is playing."


if __name__ == "__main__":
    stream = TextToAudioStream(SystemEngine())
    stream.feed(llm_text_chunks("Say hello"))
    stream.play()
```

The important rule is simple: yield strings, skip empty chunks, and let
RealtimeTTS handle sentence fragments.

## Latency Controls

For very fast first audio, start with the defaults:

```python
stream.feed(chunks).play()
```

If speech sounds choppy because the text source or TTS engine is slower than
playback, increase buffering:

```python
stream.feed(chunks).play(buffer_threshold_seconds=0.5)
```

If the first sentence starts too early for your engine, increase the first
fragment length:

```python
stream.feed(chunks).play(minimum_first_fragment_length=25)
```

For engines that sound better with complete sentences, disable early fragments:

```python
stream.feed(chunks).play(fast_sentence_fragment=False)
```

## Filtering Chunks

LLM streams often include empty events, role metadata, tool-call events, or
non-text deltas. Convert the provider response into clean strings before feeding
RealtimeTTS.

```python
def clean_text_chunks(events):
    for event in events:
        text = extract_text_somehow(event)
        if text:
            yield text
```

Keep provider-specific SDK details outside your TTS wrapper. That makes it easy
to swap the LLM provider without changing playback code.

## Lifecycle

For a one-shot streamed response:

```python
stream.feed(chunks).play()
```

For an application loop:

```python
stream.feed(chunks)
stream.play_async()

# UI, websocket, or agent loop can continue here.
```

Call `stop()` when the user interrupts playback or when a newer response should
replace the current one.

```python
stream.stop()
stream.feed(new_chunks)
stream.play_async()
```
