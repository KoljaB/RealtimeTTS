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
  - **Required**: Python >= 3.9, < 3.13
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