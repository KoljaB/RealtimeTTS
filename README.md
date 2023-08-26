# RealTimeTTS

Converts text input streams into voice audio output streams with minimal latency. Provides nearly instant auditory responses by intelligently identifying a sentence fragment from the input stream.  

This solution is perfect for applications that demand immediate and interactive audio responses.

## Features

- **Real-time Streaming**: Seamlessly stream text as you generate or input it, without waiting for the entire content.
- **Dynamic Feedback**: Ideal for applications and scenarios where immediate audio response is pivotal.
- **Modular Engine Design**: Supports custom TTS engines with a base engine provided to get you started.
- **Character-by-character Processing**: Allows for true real-time feedback as characters are read and synthesized in a stream.
- **Sentence Segmentation**: Efficiently detects sentence boundaries and synthesizes content for natural sounding output.

## Installation

```bash
pip install RealTimeTTS
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

Or you can feed character iterators for real-time streaming:

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

## Requirements

- Python 3.6+
- nltk 3.6+

## Contribution

Contributions are always welcome! Please refer to our contribution guidelines and code of conduct.

## License

MIT

## Author

Kolja Beigel  
Email: kolja.beigel@web.de  
[GitHub](https://github.com/KoljaB/RealTimeTTS)

---