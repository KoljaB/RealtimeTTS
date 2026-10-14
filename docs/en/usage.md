# Usage

## Quick Start

Here's a basic usage example:

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine() # replace with your TTS engine
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
```

## Feed Text

You can feed individual strings:

```python
stream.feed("Hello, this is a sentence.")
```

Or you can feed generators and character iterators for real-time streaming:

```python
def write(prompt: str):
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content" : prompt}],
        stream=True
    ):
        if (text_chunk := chunk["choices"][0]["delta"].get("content")) is not None:
            yield text_chunk

text_stream = write("A three-sentence relaxing speech.")

stream.feed(text_stream)
```

```python
char_iterator = iter("Streaming this character by character.")
stream.feed(char_iterator)
```

## Playback

Asynchronously:

```python
stream.play_async()
while stream.is_playing():
    time.sleep(0.1)
```

Synchronously:

```python
stream.play()
```

## Inline language controls

Enable compact language scopes with `language_tags=True`. Native QwenEngine
and QwenServerEngine implement language selection; other engines can implement
`get_language()` and `set_language(code)`. Unsupported engines raise an explicit
error rather than silently synthesizing a tagged span in another language.

`language` on TextToAudioStream remains the sentence-tokenizer setting. The
synthesis baseline comes from the engine's current language and is restored
when a scope closes, including an Auto baseline.

`language_tags` also enables the existing speed controls. Install
`realtimetts[speed]` for pitch-preserving tempo processing.

~~~python
stream = TextToAudioStream(
    engine,
    language_tags=True,
    language_voices={"es": spanish_voice},  # Optional engine voice/name.
)
stream.feed(iter(
    "Listen slowly: <es slow>Vamos al parque.</es> "
    "Now normally: <es>Vamos al parque.</es> "
    "A little faster: <es fast>Vamos al parque esta tarde.</es>"
)).play()
~~~

`<es slow>` selects Spanish at 0.75x and `<es fast>` at 1.2x. A bare language
scope uses the play operation's configured default speed. An explicit
`<es normal>` selects 1.0x. Numeric forms such as `<es speed=0.85>` and the
long form `<lang code="es">...</lang>` are also accepted. A closing tag restores
the previous language, speed and any mapped voice; scopes can nest with existing
speed and voice controls. Both angle and square brackets work.

Built-in compact codes are en, de, es, pt, fr, it, ru, zh, ja and ko. Controls
are opt-in; unknown, malformed or unfinished input remains literal. The parser
retains only a bounded tag candidate and releases body text immediately, without
waiting for a closing tag. Language boundaries also switch sentence splitting.

Without `language_voices`, the current voice remains selected and configured
Qwen server language routes still apply. With a mapping, use QwenVoice objects
for the native engine or registered server voice names for QwenServerEngine.
Native inline selection skips optional warmup synthesis; a new reference may
still need preparation, so prepare references before latency-sensitive use.
Caller-owned QwenVoice objects are not modified by inline language changes.

Completion, cancellation and errors restore the original engine state. Late
`feed()` input continues an open scope during the same play operation. Tempo
processing reaches playback, audio callbacks and WAV output through the existing
streaming path. Alignment backends must support the spoken languages separately.

## Streaming speed control

Install `realtimetts[speed]` to enable pitch-preserving speed processing. The
same output reaches speakers, `on_audio_chunk`, and WAV files.

```python
stream = TextToAudioStream(engine, speed=1.0, speed_tags=True)
stream.feed(
    'You say it like <speed factor="0.6">Worcestershire</speed>, then continue normally.'
).play()
```

`speed` accepts finite factors from 0.25 to 4.0. It is the initial/default speed
for each play operation; assigning `stream.speed` changes the next operation's
default. A closing speed tag restores the speed active before its opening tag,
including inside nested scopes.

The parser accepts these equivalent forms:

```text
<speed factor="0.6">word</speed>
<speed="0.6">word</speed>
<sr=.6>word</sr>
[speed=0.6]word[/speed]
[sr="60%"]word[/sr]
```

Built-in names are case insensitive. Single quotes, double quotes, unquoted
values, whitespace around `=`, and the `rate` or `value` attributes also work.
Opening and closing notation can be mixed: `[sr=.6]word</speed>`.
Existing unclosed `[speed=value]` tags still change subsequent text;
`[speed=reset]` restores the play operation's default. A self-closing tag,
such as `<speed value="1.2"/>`, changes the current speed without opening a scope.

Tags may arrive character by character or split across arbitrary text chunks;
the parser buffers only a bounded tag candidate, not the enclosed speech.
Both `[]` and `<>` work alongside additional delimiters configured with
`set_voice_tag_delimiters()`. If a custom pair shares an opening delimiter
with a built-in pair, the configured closing delimiter takes precedence.
Known stray closing tags are ignored. A known
closing tag that crosses nested scopes unwinds them and restores their previous
values. Unknown, invalid, oversized, or unfinished tags stay literal.
Speed markup requires `speed_tags=True`.

Registered voices and emotions also support wraps, without enabling speed
markup:

```python
stream.add_emotion("normal", normal_voice)
stream.add_emotion("whispering", whisper_voice)
stream.feed(
    '[normal]Say <style name="whispering">this quietly</style>, then normally.'
).play()
```

`<whispering>...</whispering>`, `[whispering]...[/whispering]`,
`<style="whispering">...</style>`, and the generic `voice` and `emotion`
forms resolve to the same registered mapping. Exact registered names take
precedence; case-insensitive matching works when unambiguous. Register values
accepted by your engine's `set_voice()`: these tags do not add whispering or
emotion capabilities to an engine that lacks them. The original voice is
restored from public `engine.current_voice` or `engine.voice` state when
available. For other engines, start with a registered baseline such as
`[normal]`, as above, so a closing style tag can restore it.
Registered pause tags retain their bracket forms and also accept
`<pause duration="0.2"/>`.

A speed boundary flushes preceding text for synthesis and drains the previous
audio speed segment before the next begins. Short segments can differ slightly
from the nominal duration ratio because of filter boundaries. Place tags between words, preferably
between phrases; extra synthesis boundaries can affect prosody. Normal PCM
chunks do not reset the stretcher. Generated pauses are adjusted along with
speech. Stop discards pending audio; pause/resume retains the processor state.

Compressed engines are incrementally decoded to mono float32 PCM when speed
processing is enabled. Their audio callbacks and output files then receive PCM
(int16 for callbacks/WAV), rather than the original MP3 bytes. At speed 1.0 with
tag parsing disabled, the original output path and optional dependencies remain
unchanged. Engines' own speed settings remain independent multipliers.

Existing alignment remains measured against source audio. Playback and callback
times are mapped through per-speed segments anchored to actual output duration;
retimed spans have a `retimed_` source prefix. These are derived positions, not
alignment measured on stretched audio. The controller's internal timeline stays
in source time. Tempo filtering adds lookahead, and faster playback needs the
engine to generate audio fast enough; neither inference speed nor first-audio
latency is improved by this control.

## Testing the Library

The test subdirectory contains a set of scripts to help you evaluate and understand the capabilities of the RealtimeTTS library.

Note that most of the tests still rely on the "old" OpenAI API (<1.0.0). Usage of the new OpenAI API is demonstrated in openai_1.0_test.py.

- **simple_test.py**
    - **Description**: A "hello world" styled demonstration of the library's simplest usage.

- **complex_test.py**
    - **Description**: A comprehensive demonstration showcasing most of the features provided by the library.

- **coqui_test.py**
    - **Description**: Test of local coqui TTS engine.

- **translator.py**
    - **Dependencies**: Run `pip install openai realtimestt`.
    - **Description**: Real-time translations into six different languages.

- **openai_voice_interface.py**
    - **Dependencies**: Run `pip install openai realtimestt`.
    - **Description**: Wake word activated and voice based user interface to the OpenAI API.

- **advanced_talk.py**
    - **Dependencies**: Run `pip install openai keyboard realtimestt`.
    - **Description**: Choose TTS engine and voice before starting AI conversation.

- **minimalistic_talkbot.py**
    - **Dependencies**: Run `pip install openai realtimestt`.
    - **Description**: A basic talkbot in 20 lines of code.

- **simple_llm_test.py**
    - **Dependencies**: Run `pip install openai`.
    - **Description**: Simple demonstration of how to integrate the library with large language models (LLMs).

- **test_callbacks.py**
    - **Dependencies**: Run `pip install openai`.
    - **Description**: Showcases the callbacks and lets you check the latency times in a real-world application environment.

## Pause, Resume & Stop

Pause the audio stream:

```python
stream.pause()
```

Resume a paused stream:

```python
stream.resume()
```

Stop the stream immediately:

```python
stream.stop()
```

## Requirements Explained

- **Python Version**:
  - **Required**: Python >= 3.10, < 3.15
  - **Reason**: The library depends on the GitHub library "TTS" from coqui, which requires Python versions in this range.

- **PyAudio**: to create an output audio stream

- **stream2sentence**: to split the incoming text stream into sentences

- **pyttsx3**: System text-to-speech conversion engine

- **pydub**: to convert audio chunk formats

- **azure-cognitiveservices-speech**: Azure text-to-speech conversion engine

- **elevenlabs**: Elevenlabs text-to-speech conversion engine

- **coqui-TTS**: Coqui's XTTS text-to-speech library for high-quality local neural TTS

  Shoutout to [Idiap Research Institute](https://github.com/idiap) for maintaining a [fork of coqui tts](https://github.com/idiap/coqui-ai-TTS).

- **openai**: to interact with OpenAI's TTS API

- **gtts**: Google translate text-to-speech conversion
