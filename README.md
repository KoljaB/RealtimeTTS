# RealtimeTTS

*Easy to use, low-latency text-to-speech library for realtime applications*

## About the Project

RealtimeTTS converts text streams into immediate auditory output. 

It's ideal for:

- **Voice Assistants**
- Applications requiring **instant** audio feedback

### Features

- **Realtime Streaming**: Synthesis and playback of speech as text is being generated or input
- **Sentence Segmentation**: Advanced sentence boundary detection, ensuring immediate reaction time by isolating fast-synthesizable fragments.
- **Modular Engine Design**: System TTS, Azure and Elevenlabs supported with the possibility to add custom Text-to-Speech engines

> **Hint**: *Check out [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT), the input counterpart of this library, for speech-to-text capabilities. Together, they form a powerful realtime audio wrapper around large language models.*

## Tech Stack

The library is built upon a robust and cutting-edge tech stack:

- **Text-to-Speech Engines**
  - **AzureEngine**: Microsoft's leading TTS technology. 
  - **ElevenlabsEngine**: Offer the best sounding voices available.
  - **SystemEngine**: Native engine for quick setup.

- **Sentence Boundary Detection**
  - **NLTK Sentence Tokenizer**: Uses the Natural Language Toolkit's sentence tokenizer for precise and efficient sentence segmentation.

*By using "industry standard" components RealtimeTTS offers a reliable, high-end technological foundation for developing advanced voice solutions.*

## Installation

```bash
pip install RealtimeTTS
```

## Engine Requirements

Different engines supported by RealtimeTTS have unique requirements. Ensure you fulfill these requirements based on the engine you choose.

### SystemEngine
The `SystemEngine` works out of the box using your system's built-in TTS capabilities. No additional setup is needed.

### AzureEngine
To use the `AzureEngine`, you will need:
- Microsoft Azure Text-to-Speech API key.
- Microsoft Azure service region.

Make sure you have these credentials available and correctly configured when initializing the `AzureEngine`.

### ElevenlabsEngine
For the `ElevenlabsEngine`, ensure:
- You have the Elevenlabs API key.
- `mpv` is installed on your system. It's essential for streaming audio.

  🔹 **Installing `mpv`:**
  - **macOS**:
    ```bash
    brew install mpv
    ```

  - **Linux and Windows**: Visit [mpv.io](https://mpv.io/) for installation instructions.

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

- **simple_test.py**
    - **Description**: A "hello world" styled demonstration of the library's simplest usage.

- **complex_test.py**
    - **Description**: A comprehensive demonstration showcasing most of the features provided by the library.

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
    - **Description**: Simple demonstration how to integrate the library with large language models (LLMs).

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

- Python 3.6+

- **requests (>=2.31.0)**: to send HTTP requests for API calls and voice list retrieval
  
- **PyAudio (>=0.2.13)**: to create an output audio stream
  
- **stream2sentence (>=0.1.1)**: to split the incoming text stream into sentences 

- **pyttsx3 (>=2.90)**: System text-to-speech conversion engine

- **azure-cognitiveservices-speech (>=1.31.0)**: Azure text-to-speech conversion engine
  
- **elevenlabs (>=0.2.24)**: Elevenlabs text-to-speech conversion engine


## Configuration

### Initialization Parameters for `TextToAudioStream`

When you initialize the `TextToAudioStream` class, you have various options to customize its behavior. Here are the available parameters:

#### `engine` (BaseEngine)
- **Type**: BaseEngine
- **Required**: Yes
- **Description**: The underlying engine responsible for text-to-audio synthesis. You must provide an instance of `BaseEngine` or its subclass to enable audio synthesis.

#### `on_text_stream_start` (callable)
- **Type**: Callable function
- **Required**: No
- **Description**: This optional callback function is triggered when the text stream begins. Use it for any setup or logging you may need.

#### `on_text_stream_stop` (callable)
- **Type**: Callable function
- **Required**: No
- **Description**: This optional callback function is activated when the text stream ends. You can use this for cleanup tasks or logging.

#### `on_audio_stream_start` (callable)
- **Type**: Callable function
- **Required**: No
- **Description**: This optional callback function is invoked when the audio stream starts. Useful for UI updates or event logging.

#### `on_audio_stream_stop` (callable)
- **Type**: Callable function
- **Required**: No
- **Description**: This optional callback function is called when the audio stream stops. Ideal for resource cleanup or post-processing tasks.

#### `level` (int)
- **Type**: Integer
- **Required**: No
- **Default**: `logging.WARNING`
- **Description**: Sets the logging level for the internal logger. This can be any integer constant from Python's built-in `logging` module.

#### Example Usage:

```python
engine = YourEngine()  # Substitute with your engine
stream = TextToAudioStream(
    engine=engine,
    on_text_stream_start=my_text_start_func,
    on_text_stream_stop=my_text_stop_func,
    on_audio_stream_start=my_audio_start_func,
    on_audio_stream_stop=my_audio_stop_func,
    level=logging.INFO
)
```

### Methods

#### `play` and `play_async`

These methods are responsible for executing the text-to-audio synthesis and playing the audio stream. The difference is that `play` is a blocking function, while `play_async` runs in a separate thread, allowing other operations to proceed.

##### `fast_sentence_fragment` (bool)
- **Default**: `False`
- **Description**: When set to `True`, the method will prioritize speed, generating and playing sentence fragments faster. This is useful for applications where latency matters.

##### `buffer_threshold_seconds` (float)
- **Default**: `2.0`
- **Description**: Specifies the time in seconds for the buffering threshold, which impacts the smoothness and continuity of audio playback. 

  - **How it Works**: Before synthesizing a new sentence, the system checks if there is more audio material left in the buffer than the time specified by `buffer_threshold_seconds`. If so, it retrieves another sentence from the text generator, assuming that it can fetch and synthesize this new sentence within the time window provided by the remaining audio in the buffer. This process allows the text-to-speech engine to have more context for better synthesis, enhancing the user experience.

  A higher value ensures that there's more pre-buffered audio, reducing the likelihood of silence or gaps during playback. If you experience breaks or pauses, consider increasing this value.

- **Hint**: If you experience silence or breaks between sentences, consider raising this value to ensure smoother playback.

##### `minimum_sentence_length` (int)
- **Default**: `3`
- **Description**: Sets the minimum character length to consider a string as a sentence to be synthesized. This affects how text chunks are processed and played.

##### `log_characters` (bool)
- **Default**: `False`
- **Description**: Enable this to log the individual characters that are being processed for synthesis.

##### `log_synthesized_text` (bool)
- **Default**: `False`
- **Description**: When enabled, logs the text chunks as they are synthesized into audio. Helpful for auditing and debugging.

By understanding and setting these parameters and methods appropriately, you can tailor the `TextToAudioStream` to meet the specific needs of your application.

## Contribution

Contributions are always welcome (e.g. PR to add a new engine).

## License

MIT

## Author

Kolja Beigel  
Email: kolja.beigel@web.de  
[GitHub](https://github.com/KoljaB/RealtimeTTS)