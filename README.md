# RealtimeTTS
[![PyPI](https://img.shields.io/pypi/v/RealtimeTTS)](https://pypi.org/project/RealtimeTTS/)
[![Downloads](https://static.pepy.tech/badge/RealtimeTTS)](https://pepy.tech/project/KoljaB/RealtimeTTS)
[![GitHub release](https://img.shields.io/github/release/KoljaB/RealtimeTTS.svg)](https://GitHub.com/KoljaB/RealtimeTTS/releases/)
[![GitHub commits](https://badgen.net/github/commits/KoljaB/RealtimeTTS)](https://GitHub.com/Naereen/KoljaB/RealtimeTTS/commit/)
[![GitHub forks](https://img.shields.io/github/forks/KoljaB/RealtimeTTS.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/KoljaB/RealtimeTTS/network/)
[![GitHub stars](https://img.shields.io/github/stars/KoljaB/RealtimeTTS.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/KoljaB/RealtimeTTS/stargazers/)

*Easy to use, low-latency text-to-speech library for realtime applications*

## About the Project

RealtimeTTS is a state-of-the-art text-to-speech (TTS) library designed for real-time applications. It stands out in its ability to convert text streams fast into high-quality auditory output with minimal latency.

> **Important:** [Installation](#installation) has changed to allow more customization. Please use `pip install realtimetts[all]` instead of `pip install realtimetts` now. More [info here](#installation).

> **Hint:** *<strong>Check out [Linguflex](https://github.com/KoljaB/Linguflex)</strong>, the original project from which RealtimeTTS is spun off. It lets you control your environment by speaking and is one of the most capable and sophisticated open-source assistants currently available.*

https://github.com/KoljaB/RealtimeTTS/assets/7604638/87dcd9a5-3a4e-4f57-be45-837fc63237e7

## Key Features

- **Low Latency**
  - almost instantaneous text-to-speech conversion
  - compatible with LLM outputs
- **High-Quality Audio**
  - generates clear and natural-sounding speech
- **Multiple TTS Engine Support**
  - supports OpenAI TTS, Elevenlabs, Azure Speech Services, Coqui TTS, gTTS and System TTS
- **Multilingual**
- **Robust and Reliable**:
  - ensures continuous operation through a fallback mechanism
  - switches to alternative engines in case of disruptions guaranteeing consistent performance and reliability, which is vital for critical and professional use cases

> **Hint**: *check out [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT), the input counterpart of this library, for speech-to-text capabilities. Together, they form a powerful realtime audio wrapper around large language models.*

## FAQ

Check the [FAQ page](./FAQ.md) for answers to a lot of questions around the usage of RealtimeTTS.

## Documentation

The documentation for **RealtimeTTS** is available in both English and French. 

- **[English Documentation](https://koljab.github.io/RealtimeTTS/en/)**
- **[French Documentation](https://koljab.github.io/RealtimeTTS/fr/)**

## Updates

Latest Version: v0.4.7

See [release history](https://github.com/KoljaB/RealtimeTTS/releases).

## Tech Stack

This library uses:

- **Text-to-Speech Engines**
  - **OpenAIEngine**: OpenAI's TTS system offers 6 natural sounding voices.
  - **CoquiEngine**: High quality local neural TTS.
  - **AzureEngine**: Microsoft's leading TTS technology. 500,000 chars free per month.
  - **ElevenlabsEngine**: Offers the best sounding voices available.
  - **GTTSEngine**: Free to use and doesn't require setting up a local GPU.
  - **SystemEngine**: Native engine for quick setup.

- **Sentence Boundary Detection**
  - **NLTK Sentence Tokenizer**: Natural Language Toolkit's sentence tokenizer for straightforward text-to-speech tasks in English or when simplicity is preferred.
  - **Stanza Sentence Tokenizer**: Stanza sentence tokenizer for working with multilingual text or when higher accuracy and performance are required.

*By using "industry standard" components RealtimeTTS offers a reliable, high-end technological foundation for developing advanced voice solutions.*

## Installation

> **Note:** Basic Installation with `pip install realtimetts` is not recommended anymore, use `pip install realtimetts[all]` instead.

The RealtimeTTS library provides installation options for various dependencies for your use case. Here are the different ways you can install RealtimeTTS depending on your needs:

### Full Installation

To install RealtimeTTS with support for all TTS engines:

```bash
pip install -U realtimetts[all]
```

### Custom Installation

RealtimeTTS allows for custom installation with minimal library installations. Here are the options available:
- **all**: Full installation with every engine supported.
- **system**: Includes system-specific TTS capabilities (e.g., pyttsx3).
- **azure**: Adds Azure Cognitive Services Speech support.
- **elevenlabs**: Includes integration with ElevenLabs API.
- **openai**: For OpenAI voice services.
- **gtts**: Google Text-to-Speech support.
- **coqui**: Installs the Coqui TTS engine.
- **minimal**: Installs only the base requirements with no engine (only needed if you want to develop an own engine)


Say you want to install RealtimeTTS only for local neuronal Coqui TTS usage, then you should use:

```bash
pip install realtimetts[coqui]
```

For example, if you want to install RealtimeTTS with only Azure Cognitive Services Speech, ElevenLabs, and OpenAI support:

```bash
pip install realtimetts[azure,elevenlabs,openai]
```

### Virtual Environment Installation

For those who want to perform a full installation within a virtual environment, follow these steps:

```bash
python -m venv env_realtimetts
env_realtimetts\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install -U realtimetts[all]
```

More information about [CUDA installation](#cuda-installation).

## Engine Requirements

Different engines supported by RealtimeTTS have unique requirements. Ensure you fulfill these requirements based on the engine you choose.

### SystemEngine
The `SystemEngine` works out of the box with your system's built-in TTS capabilities. No additional setup is needed.

### GTTSEngine
The `GTTSEngine` works out of the box using Google Translate's text-to-speech API. No additional setup is needed.

### OpenAIEngine
To use the `OpenAIEngine`:
- set environment variable OPENAI_API_KEY
- install ffmpeg (see [CUDA installation](#cuda-installation) point 3)

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

Downloads a neural TTS model first. In most cases it be fast enough for Realtime using GPU synthesis. Needs around 4-5 GB VRAM.

- to clone a voice submit the filename of a wave file containing the source voice as "voice" parameter to the CoquiEngine constructor
- voice cloning works best with a 22050 Hz mono 16bit WAV file containing a short (~5-30 sec) sample

On most systems GPU support will be needed to run fast enough for realtime, otherwise you will experience stuttering.

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

#### `output_device_index` (int)
- **Type**: Integer
- **Required**: No
- **Default**: None
- **Description**: Specifies the output device index to use. None uses the default device.

#### `tokenizer` (string)
- **Type**: String
- **Required**: No
- **Default**: nltk
- **Description**: Tokenizer to use for sentence splitting (currently "nltk" and "stanza" are supported).

#### `language` (string)
- **Type**: String
- **Required**: No
- **Default**: en
- **Description**: Language to use for sentence splitting.

#### `muted` (bool)
- **Type**: Bool
- **Required**: No
- **Default**: False
- **Description**: Global muted parameter. If True, no pyAudio stream will be opened. Disables audio playback via local speakers (in case you want to synthesize to file or process audio chunks) and overrides the play parameters muted setting.

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

##### Parameters:

###### `fast_sentence_fragment` (bool)
- **Default**: `True`
- **Description**: When set to `True`, the method will prioritize speed, generating and playing sentence fragments faster. This is useful for applications where latency matters.

###### `fast_sentence_fragment_allsentences` (bool)
- **Default**: `False`
- **Description**: When set to `True`, applies the fast sentence fragment processing to all sentences, not just the first one.

###### `fast_sentence_fragment_allsentences_multiple` (bool)
- **Default**: `False`
- **Description**: When set to `True`, allows yielding multiple sentence fragments instead of just a single one.

###### `buffer_threshold_seconds` (float)
- **Default**: `0.0`
- **Description**: Specifies the time in seconds for the buffering threshold, which impacts the smoothness and continuity of audio playback.

  - **How it Works**: Before synthesizing a new sentence, the system checks if there is more audio material left in the buffer than the time specified by `buffer_threshold_seconds`. If so, it retrieves another sentence from the text generator, assuming that it can fetch and synthesize this new sentence within the time window provided by the remaining audio in the buffer. This process allows the text-to-speech engine to have more context for better synthesis, enhancing the user experience.

  A higher value ensures that there's more pre-buffered audio, reducing the likelihood of silence or gaps during playback. If you experience breaks or pauses, consider increasing this value.

###### `minimum_sentence_length` (int)
- **Default**: `10`
- **Description**: Sets the minimum character length to consider a string as a sentence to be synthesized. This affects how text chunks are processed and played.

###### `minimum_first_fragment_length` (int)
- **Default**: `10`
- **Description**: The minimum number of characters required for the first sentence fragment before yielding.

###### `log_synthesized_text` (bool)
- **Default**: `False`
- **Description**: When enabled, logs the text chunks as they are synthesized into audio. Helpful for auditing and debugging.

###### `reset_generated_text` (bool)
- **Default**: `True`
- **Description**: If True, reset the generated text before processing.

###### `output_wavfile` (str)
- **Default**: `None`
- **Description**: If set, save the audio to the specified WAV file.

###### `on_sentence_synthesized` (callable)
- **Default**: `None`
- **Description**: A callback function that gets called after a single sentence fragment was synthesized.

###### `before_sentence_synthesized` (callable)
- **Default**: `None`
- **Description**: A callback function that gets called before a single sentence fragment gets synthesized.

###### `on_audio_chunk` (callable)
- **Default**: `None`
- **Description**: Callback function that gets called when a single audio chunk is ready.

###### `tokenizer` (str)
- **Default**: `"nltk"`
- **Description**: Tokenizer to use for sentence splitting. Currently supports "nltk" and "stanza".

###### `tokenize_sentences` (callable)
- **Default**: `None`
- **Description**: A custom function that tokenizes sentences from the input text. You can provide your own lightweight tokenizer if you are unhappy with nltk and stanza. It should take text as a string and return split sentences as a list of strings.

###### `language` (str)
- **Default**: `"en"`
- **Description**: Language to use for sentence splitting.

###### `context_size` (int)
- **Default**: `12`
- **Description**: The number of characters used to establish context for sentence boundary detection. A larger context improves the accuracy of detecting sentence boundaries.

###### `context_size_look_overhead` (int)
- **Default**: `12`
- **Description**: Additional context size for looking ahead when detecting sentence boundaries.

###### `muted` (bool)
- **Default**: `False`
- **Description**: If True, disables audio playback via local speakers. Useful when you want to synthesize to a file or process audio chunks without playing them.

###### `sentence_fragment_delimiters` (str)
- **Default**: `".?!;:,\n…)]}。-"`
- **Description**: A string of characters that are considered sentence delimiters.

###### `force_first_fragment_after_words` (int)
- **Default**: `15`
- **Description**: The number of words after which the first sentence fragment is forced to be yielded.

### CUDA installation

These steps are recommended for those who require **better performance** and have a compatible NVIDIA GPU.

> **Note**: *to check if your NVIDIA GPU supports CUDA, visit the [official CUDA GPUs list](https://developer.nvidia.com/cuda-gpus).*

To use a torch with support via CUDA please follow these steps:

> **Note**: *newer pytorch installations [may](https://stackoverflow.com/a/77069523) (unverified) not need Toolkit (and possibly cuDNN) installation anymore.*

1. **Install NVIDIA CUDA Toolkit**:
    For example, to install Toolkit 12.X, please
    - Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).
    - Select your operating system, system architecture, and os version.
    - Download and install the software.

    or to install Toolkit 11.8, please
    - Visit [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).
    - Select your operating system, system architecture, and os version.
    - Download and install the software.

2. **Install NVIDIA cuDNN**:

    For example, to install cuDNN 8.7.0 for CUDA 11.x please
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

    To upgrade your PyTorch installation to enable GPU support with CUDA, follow these instructions based on your specific CUDA version. This is useful if you wish to enhance the performance of RealtimeSTT with CUDA capabilities.

    - **For CUDA 11.8:**

        To update PyTorch and Torchaudio to support CUDA 11.8, use the following commands:

        ```bash
        pip install torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
        ```

    - **For CUDA 12.X:**


        To update PyTorch and Torchaudio to support CUDA 12.X, execute the following:

        ```bash
        pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
        ```

    Replace `2.3.1` with the version of PyTorch that matches your system and requirements.

5. **Fix for to resolve compatibility issues**:
    If you run into library compatibility issues, try setting these libraries to fixed versions:

    ```bash
    pip install networkx==2.8.8
    pip install typing_extensions==4.8.0
    pip install fsspec==2023.6.0
    pip install imageio==2.31.6
    pip install networkx==2.8.8
    pip install numpy==1.24.3
    pip install requests==2.31.0
    ```

## 💖 Acknowledgements

Huge shoutout to the team behind [Coqui AI](https://coqui.ai/) - especially the brilliant [Eren Gölge](https://github.com/erogol) - for being the first to give us local high-quality synthesis with real-time speed and even a clonable voice!

Thank you [Pierre Nicolas Durette](https://github.com/pndurette) for giving us a free tts to use without GPU using Google Translate with his gtts python library.

## Contribution

Contributions are always welcome (e.g. PR to add a new engine).

## License Information

### ❗ Important Note:
While the source of this library is open-source, the usage of many of the engines it depends on is not: External engine providers often restrict commercial use in their free plans. This means the engines can be used for noncommercial projects, but commercial usage requires a paid plan.

### Engine Licenses Summary:

#### CoquiEngine
- **License**: Open-source only for noncommercial projects.
- **Commercial Use**: Requires a paid plan.
- **Details**: [CoquiEngine License](https://coqui.ai/cpml)

#### ElevenlabsEngine
- **License**: Open-source only for noncommercial projects.
- **Commercial Use**: Available with every paid plan.
- **Details**: [ElevenlabsEngine License](https://help.elevenlabs.io/hc/en-us/articles/13313564601361-Can-I-publish-the-content-I-generate-on-the-platform-)

#### AzureEngine
- **License**: Open-source only for noncommercial projects.
- **Commercial Use**: Available from the standard tier upwards.
- **Details**: [AzureEngine License](https://learn.microsoft.com/en-us/answers/questions/1192398/can-i-use-azure-text-to-speech-for-commercial-usag)

#### SystemEngine
- **License**: Mozilla Public License 2.0 and GNU Lesser General Public License (LGPL) version 3.0.
- **Commercial Use**: Allowed under this license.
- **Details**: [SystemEngine License](https://github.com/nateshmbhat/pyttsx3/blob/master/LICENSE)

#### GTTSEngine
- **License**: MIT license
- **Commercial Use**: It's under the MIT license, so it should be theoretically possible. Some caution might be necessary since it utilizes undocumented Google Translate speech functionality.
- **Details**: [GTTS MIT License](https://github.com/pndurette/gTTS/blob/main/LICENSE)

#### OpenAIEngine
- **License**: please read [OpenAI Terms of Use](https://openai.com/policies/terms-of-use)

**Disclaimer**: This is a summarization of the licenses as understood at the time of writing. It is not legal advice. Please read and respect the licenses of the different engine providers if you plan to use them in a project.

## Contributors

<a href="https://github.com/traceloop/openllmetry/graphs/contributors">
  <img alt="contributors" src="https://contrib.rocks/image?repo=KoljaB/RealtimeTTS"/>
</a>

## Author

Kolja Beigel
Email: kolja.beigel@web.de



<p align="center">
  <a href="https://github.com/KoljaB/RealtimeTTS" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="#realtimetts" target="_blank">
    <img src="https://img.shields.io/badge/Back%20to%20Top-000000?style=for-the-badge" alt="Back to Top">
  </a>
</p>
