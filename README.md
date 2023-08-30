# RealtimeTTS

Converts text input streams into voice audio output streams with minimal latency. Provides nearly instant auditory responses by quickly identifying a sentence fragment from the input stream.  

This solution is perfect for applications that demand immediate and interactive audio responses.

> **Hint**: Looking for a way to convert voice audio input into text? Check out [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT), the perfect input counterpart for this library. Together, they form a powerful realtime audio wrapper around large language model outputs.

## Features

- **Real-time Streaming**: Stream text as you generate or input it, without waiting for the entire content.
- **Dynamic Feedback**: Ideal for applications and scenarios where immediate audio response is pivotal.
- **Modular Engine Design**: Supports custom TTS engines with system tts, azure and elevenlabs engines provided to get you started.
- **Character-by-character Processing**: Allows for true real-time feedback as characters are read and synthesized in a stream.
- **Sentence Segmentation**: Efficiently detects sentence boundaries and synthesizes content for natural sounding output.

## Installation

```bash
pip install RealtimeTTS
```

## Quick Start

Here's a basic usage example:

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine() # replace with your TTS engine
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
```

## Usage

### Feed Text

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

### Playback

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

### Tests

In the tests directory there are some files to test the library.  

simple_test.py, complex_test.py will work out of the box.
simple_llm_test.py needs openai library installed (pip install openai).
simple_talk.py, advanced_talk.py need faster_whisper, torch and keyboard library (pip install openai keyboard, faster_whisper, torch).
translator_cli.py needs RealtimeSST library to work


### Pause, Resume & Stop

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

- Python 3.6+

- **requests (>=2.31.0)**: to send HTTP requests for API calls and voice list retrieval
  
- **PyAudio (>=0.2.13)**: to create an output audio stream
  
- **stream2sentence (>=0.1.1)**: to split the incoming text stream into sentences 

- **pyttsx3 (>=2.90)**: System text-to-speech conversion engine

- **azure-cognitiveservices-speech (>=1.31.0)**: Azure text-to-speech conversion engine
  
- **elevenlabs (>=0.2.24)**: Elevenlabs text-to-speech conversion engine

## Contribution

Contributions are always welcome (e.g. PR to add a new engine).

## License

MIT

## Author

Kolja Beigel  
Email: kolja.beigel@web.de  
[GitHub](https://github.com/KoljaB/RealtimeTTS)

---