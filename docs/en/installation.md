
> **Note:** Basic Installation with `pip install realtimetts` is not recommended anymore, use `pip install realtimetts[all]` instead.

The RealtimeTTS library provides installation options for various dependencies for your use case. Here are the different ways you can install RealtimeTTS depending on your needs:

### Full Installation

To install RealtimeTTS with support for all TTS engines:

```
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

```
pip install realtimetts[coqui]
```

For example, if you want to install RealtimeTTS with only Azure Cognitive Services Speech, ElevenLabs, and OpenAI support:

```
pip install realtimetts[azure,elevenlabs,openai]
```

### Virtual Environment Installation

For those who want to perform a full installation within a virtual environment, follow these steps:

```
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
    ```
    brew install mpv
    ```

  - **Linux and Windows**: Visit [mpv.io](https://mpv.io/) for installation instructions.

### CoquiEngine

Delivers high quality, local, neural TTS with voice-cloning.

Downloads a neural TTS model first. In most cases it be fast enough for Realtime using GPU synthesis. Needs around 4-5 GB VRAM.

- to clone a voice submit the filename of a wave file containing the source voice as "voice" parameter to the CoquiEngine constructor
- voice cloning works best with a 22050 Hz mono 16bit WAV file containing a short (~5-30 sec) sample

On most systems GPU support will be needed to run fast enough for realtime, otherwise you will experience stuttering.

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
        ```
        sudo apt update && sudo apt install ffmpeg
        ```

    - **On Arch Linux**:
        ```
        sudo pacman -S ffmpeg
        ```

    - **On MacOS using Homebrew** ([https://brew.sh/](https://brew.sh/)):
        ```
        brew install ffmpeg
        ```

    - **On Windows using Chocolatey** ([https://chocolatey.org/](https://chocolatey.org/)):
        ```
        choco install ffmpeg
        ```

    - **On Windows using Scoop** ([https://scoop.sh/](https://scoop.sh/)):
        ```
        scoop install ffmpeg
        ```

4. **Install PyTorch with CUDA support**:

    To upgrade your PyTorch installation to enable GPU support with CUDA, follow these instructions based on your specific CUDA version. This is useful if you wish to enhance the performance of RealtimeSTT with CUDA capabilities.

    - **For CUDA 11.8:**

        To update PyTorch and Torchaudio to support CUDA 11.8, use the following commands:

        ```
        pip install torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
        ```

    - **For CUDA 12.X:**


        To update PyTorch and Torchaudio to support CUDA 12.X, execute the following:

        ```
        pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
        ```

    Replace `2.3.1` with the version of PyTorch that matches your system and requirements.

5. **Fix for to resolve compatibility issues**:
    If you run into library compatibility issues, try setting these libraries to fixed versions:

  ``` 

    pip install networkx==2.8.8
    
    pip install typing_extensions==4.8.0
    
    pip install fsspec==2023.6.0
    
    pip install imageio==2.31.6
    
    pip install networkx==2.8.8
    
    pip install numpy==1.24.3
    
    pip install requests==2.31.0
  ```