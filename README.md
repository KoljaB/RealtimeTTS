# RealtimeTTS

*Easy to use, low-latency text-to-speech library for realtime applications*

## About the Project

RealtimeTTS is a state-of-the-art text-to-speech (TTS) library designed for real-time applications. It stands out in its ability to convert text streams into high-quality auditory output with minimal latency. This makes it an ideal solution for voice assistants, interactive games, and any applications requiring instant audio feedback.

https://github.com/KoljaB/RealtimeTTS/assets/7604638/87dcd9a5-3a4e-4f57-be45-837fc63237e7


### Key Features

- **Low Latency**
  - provides almost instantaneous text-to-speech conversion
  - compatible with LLM outputs
- **High-Quality Audio**
  - generates clear and natural-sounding speech
- **Multilingual and Multiple Engine Support**
  - supports various languages and TTS engines
- **Robust and Reliable**: 
  - ensures continuous operation with a fallback mechanism
  - switches to alternative engines in case of disruptions guaranteeing consistent performance and reliability, which is vital for critical and professional use cases

> **Hint**: *Check out [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT), the input counterpart of this library, for speech-to-text capabilities. Together, they form a powerful realtime audio wrapper around large language models.*

### Updates

Latest Version: v0.3.0

#### New Features:
1. Expanded language support, including Chinese (details in [tests](https://github.com/KoljaB/RealtimeTTS/blob/master/tests/chinese_test.py) and [speed test](https://github.com/KoljaB/RealtimeTTS/blob/master/tests/pyqt6_speed_test_chinese.py)).
2. Fallback engines in TextToAudioStream, enhancing reliability for real-time scenarios by switching to alternate engines if one fails.
3. Audio file saving feature with `output_wavfile` parameter. This allows for the simultaneous saving of real-time synthesized audio, enabling later playback of the live synthesis.

For more details, see the [release history](https://github.com/KoljaB/RealtimeTTS/releases).

## Tech Stack

This library uses:

- **Text-to-Speech Engines**
  - **CoquiEngine**: High quality local neural TTS.
  - **AzureEngine**: Microsoft's leading TTS technology. 250000 chars free per month.
  - **ElevenlabsEngine**: Offer the best sounding voices available.
  - **SystemEngine**: Native engine for quick setup.

- **Sentence Boundary Detection**
  - **NLTK Sentence Tokenizer**: Uses the Natural Language Toolkit's sentence tokenizer for precise and efficient sentence segmentation.

*By using "industry standard" components RealtimeTTS offers a reliable, high-end technological foundation for developing advanced voice solutions.*

## Installation

```bash
pip install RealtimeTTS
```

This will install all the necessary dependencies, including a **CPU support only** version of PyTorch (needed for Coqui engine)

To use Coqui engine it is recommended to upgrade torch to GPU usage (see installation steps under [Coqui Engine](#coquiengine) further below).

## Engine Requirements

Different engines supported by RealtimeTTS have unique requirements. Ensure you fulfill these requirements based on the engine you choose.

### SystemEngine
The `SystemEngine` works out of the box using your system's built-in TTS capabilities. No additional setup is needed.

### AzureEngine
To use the `AzureEngine`, you will need:
- Microsoft Azure Text-to-Speech API key (provided via AzureEngine constructor parameter "speech_key" or in the environment variable AZURE_SPEECH_KEY)
- Microsoft Azure service region.

Make sure you have these credentials available and correctly configured when initializing the `AzureEngine`.

### ElevenlabsEngine
For the `ElevenlabsEngine`, you need:
- Elevenlabs API key (provided via ElevenlabsEngine constructor parameter "api_key" or in the environment variable ELEVENLABS_API_KEY)
- `mpv` installed on your system (essential for streaming mpeg audio, Elevenlabs only delivers mpeg).

  🔹 **Installing `mpv`:**
  - **macOS**:
    ```bash
    brew install mpv
    ```

  - **Linux and Windows**: Visit [mpv.io](https://mpv.io/) for installation instructions.

### CoquiEngine

Delivers high quality, local, neural TTS with voice-cloning.  

Downloads a neural TTS model first. In most cases it be fast enought for Realtime using GPU synthesis. Needs around 4-5 GB VRAM.

- to clone a voice submit the filename of a wave file containing the source voice as cloning_reference_wav to the CoquiEngine constructor
- in my experience voice cloning works best with a 24000 Hz mono 16bit WAV file containing a short (~10 sec) sample 

#### GPU-Support (CUDA) for Coqui

Additional steps are needed for a **GPU-optimized** installation. These steps are recommended for those who require **better performance** and have a compatible NVIDIA GPU.

> **Note**: *To check if your NVIDIA GPU supports CUDA, visit the [official CUDA GPUs list](https://developer.nvidia.com/cuda-gpus).*

To use local Coqui Engine with GPU support via CUDA please follow these steps:

1. **Install NVIDIA CUDA Toolkit 11.8**:
    - Visit [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).
    - Select version 11.
    - Download and install the software.

2. **Install NVIDIA cuDNN 8.7.0 for CUDA 11.x**:
    - Visit [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
    - Click on "Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x".
    - Download and install the software.

3. **Install ffmpeg**:

    You can download an installer for your OS from the [ffmpeg Website](https://ffmpeg.org/download.html).  
    
    Or use a package manager:

    - **On Ubuntu or Debian**:
        ```bash
        sudo apt update && sudo apt install ffmpeg
        ```

    - **On Arch Linux**:
        ```bash
        sudo pacman -S ffmpeg
        ```

    - **On MacOS using Homebrew** ([https://brew.sh/](https://brew.sh/)):
        ```bash
        brew install ffmpeg
        ```

    - **On Windows using Chocolatey** ([https://chocolatey.org/](https://chocolatey.org/)):
        ```bash
        choco install ffmpeg
        ```

    - **On Windows using Scoop** ([https://scoop.sh/](https://scoop.sh/)):
        ```bash
        scoop install ffmpeg
        ```    

4. **Install PyTorch with CUDA support**:
    ```bash
    pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
    ```

5. **Fix for to resolve compatility issues**:
    If you run into library compatility issues, please set these libraries to fixed versions:

    ```bash
    pip install networkx==2.8.8
    pip install typing_extensions==4.8.0
    pip install fsspec==2023.6.0
    pip install imageio==2.31.6
    pip install networkx==2.8.8
    pip install numpy==1.24.3
    pip install requests==2.31.0
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

#### `on_character` (callable)
- **Type**: Callable function
- **Required**: No
- **Description**: This optional callback function is called when a single character is processed.

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

❗
While the source of this library is under MIT, some libraries it depends on are not.  
A lot of external engine providers currently DO NOT ALLOW commercial use together with their free plans.  
Please read and respect the licenses of the different engine providers.

[CoquiEngine](https://coqui.ai/cpml)
- non-commercial for free plan, commercial paid plans available
	
[ElevenlabsEngine](https://help.elevenlabs.io/hc/en-us/articles/13313564601361-Can-I-publish-the-content-I-generate-on-the-platform-)
- non-commercial for free plan, commercial for every paid plan  

[AzureEngine](https://learn.microsoft.com/en-us/answers/questions/1192398/can-i-use-azure-text-to-speech-for-commercial-usag)
- non-commercial for free tier, commercial for standard tier upwards  
	  
SystemEngine:  
- GNU Lesser General Public License (LGPL) version 3.0  
- commercial use allowed

## Author

Kolja Beigel  
Email: kolja.beigel@web.de  
[GitHub](https://github.com/KoljaB/RealtimeTTS)
