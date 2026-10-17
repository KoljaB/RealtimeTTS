
> **注意：** 不再推荐使用 `pip install realtimetts` 进行基本安装，而应使用 `pip install realtimetts[all]`。

RealtimeTTS 库为您的用例提供了各种依赖项的安装选项。以下是根据需要安装 RealtimeTTS 的不同方法：

### 完整安装

安装支持所有 TTS 引擎的 RealtimeTTS：

```
pip install -U realtimetts[all]
```

### 定制安装

RealtimeTTS 允许使用最少的库安装进行自定义安装。以下是可用选项：
- 全部***： 完全安装所有支持的引擎。
- **系统**： 包括特定系统的 TTS 功能（如 pyttsx3）。
- **azure**： 添加 Azure 认知服务语音支持。
- **elevenlabs**： 包括与 ElevenLabs API 的集成。
- **openai**： 用于 OpenAI 语音服务。
- **gtts**： 支持谷歌文本到语音。
- **coqui**： 安装 Coqui TTS 引擎。
- **minimal**： 只安装基本要求，不安装引擎（只有当你想开发自己的引擎时才需要）。


如果您只想为本地神经元 Coqui TTS 安装 RealtimeTTS，则应使用

```
pip install realtimetts[coqui］
```

例如，如果您想安装只支持 Azure 认知服务语音、ElevenLabs 和 OpenAI 的 RealtimeTTS：

```
pip install realtimetts[azure,elevenlabs,openai].
```

### 虚拟环境安装

如果想在虚拟环境中进行完整安装，请按照以下步骤操作：

```
python -m venv env_realtimetts
env_realtimetts\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install -U realtimetts[all]
```

有关 [CUDA 安装](#cuda-installation) 的更多信息。

## 引擎要求

RealtimeTTS 支持的不同引擎有其独特的要求。请根据所选引擎确保满足这些要求。

### 系统引擎
系统引擎 "开箱即用，具有系统内置的 TTS 功能。无需额外设置。

### GTTS 引擎
GTTSEngine "开箱即用，使用谷歌翻译的文本到语音 API。无需额外设置。

### OpenAIEngine
要使用 `OpenAIEngine`：
- 设置环境变量 OPENAI_API_KEY
- 安装 `realtimetts[openai]`（包含用于 MP3 播放的 PyAV）。

### AzureEngine
要使用 “AzureEngine”，你需要
- Microsoft Azure 文本到语音 API 密钥（通过 AzureEngine 构造函数参数 “speech_key ”或环境变量 AZURE_SPEECH_KEY 提供）
- Microsoft Azure 服务区域。

在初始化 `AzureEngine` 时，确保这些凭据可用并配置正确。

### ElevenlabsEngine
使用 `ElevenlabsEngine` 时需要
- Elevenlabs API 密钥（通过 ElevenlabsEngine 构造函数参数 “api_key ”或环境变量 ELEVENLABS_API_KEY 提供）
- 系统中已安装 `mpv`（用于流式传输 mpeg 音频，Elevenlabs 仅提供 mpeg）。

  🔹 **安装 `mpv`:**
  - macOS**：
    ```
    brew install mpv
    ```

  - **Linux和Windows**： 请访问 [mpv.io](https://mpv.io/) 获取安装说明。

#### CoquiEngine

通过语音克隆提供高质量的本地神经 TTS。

首先下载一个神经 TTS 模型。在大多数情况下，使用 GPU 合成的实时速度足够快。需要大约 4-5GB VRAM。

- 要克隆语音，请将包含源语音的波形文件的文件名作为 “语音 ”参数提交给 CoquiEngine 构造函数
- 语音克隆最好使用 22050 Hz 单声道 16 位 WAV 文件，其中包含一个短（约 5-30 秒）样本

在大多数系统上，需要 GPU 的支持才能以足够快的速度实时运行，否则会出现卡顿现象。

### CUDA 安装

这些步骤适用于那些需要**更好性能**并且拥有兼容的NVIDIA GPU的人。

> **注意**：*要检查您的NVIDIA GPU是否支持CUDA，请访问[官方CUDA GPU列表](https://developer.nvidia.com/cuda-gpus)。*

要使用支持CUDA的torch，请按照以下步骤操作：

> **注意**：*较新的 PyTorch 安装 [可能](https://stackoverflow.com/a/77069523)（未经验证）不再需要安装 Toolkit（可能也不需要安装 cuDNN）。*

1. **安装 NVIDIA CUDA Toolkit**：
    例如，要安装 Toolkit 12.X，请
    - 访问 [NVIDIA CUDA 下载](https://developer.nvidia.com/cuda-downloads)。
    - 选择你的操作系统、系统架构和操作系统版本。
    - 下载并安装软件。

    或者要安装 Toolkit 11.8，请
    - 访问 [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)。
    - 选择您的操作系统、系统架构和操作系统版本。
    - 下载并安装软件。

2. **安装 NVIDIA cuDNN**：

    例如，要为CUDA 11.x安装cuDNN 8.7.0，请
    - 访问[NVIDIA cuDNN归档](https://developer.nvidia.com/rdp/cudnn-archive)。
    - 点击“下载 cuDNN v8.7.0（2022年11月28日），适用于 CUDA 11.x”。
    - 下载并安装软件。

3. **安装 ffmpeg**：

    您可以从 [ffmpeg 网站](https://ffmpeg.org/download.html) 下载适用于您操作系统的安装程序。

    或者使用包管理器：

    - **在 Ubuntu 或 Debian 上**：
        ```
        sudo apt update && sudo apt install ffmpeg
        ```

    - **在 Arch Linux 上**：
        ```
        sudo pacman -S ffmpeg
        ```

    - **在使用 Homebrew 的 MacOS 上** ([https://brew.sh/](https://brew.sh/)):
        ``` 
        brew install ffmpeg
        ```

    - **在Windows上使用Chocolatey** ([https://chocolatey.org/](https://chocolatey.org/)):
        ```
        choco install ffmpeg
```

    - **在Windows上使用Scoop** ([https://scoop.sh/](https://scoop.sh/)):
        ```plaintext
        scoop install ffmpeg
        ```

4. **安装带有CUDA支持的PyTorch**：

    要升级您的PyTorch安装以启用CUDA的GPU支持，请根据您的具体CUDA版本遵循以下说明。 如果您希望通过CUDA功能提升RealtimeSTT的性能，这将非常有用。

    - **对于CUDA 11.8：**

        要更新 PyTorch 和 Torchaudio 以支持 CUDA 11.8，请使用以下命令：

        ```
        pip install torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```
        文本待翻译：```

    - **对于CUDA 12.X：**


        要更新 PyTorch 和 Torchaudio 以支持 CUDA 12.X，请执行以下操作：

        ```plaintext
pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
        文本待翻译：```

    将 `2.3.1` 替换为适合您的系统和需求的 PyTorch 版本。

5. **修复以解决兼容性问题**：
    如果你遇到库兼容性问题，尝试将这些库设置为固定版本：

  文本翻译：``` 

    pip install networkx==2.8.8
    
    pip install typing_extensions==4.8.0
    
    pip install fsspec==2023.6.0
    
    pip install imageio==2.31.6
    
    pip install networkx==2.8.8
    
    pip install numpy==1.24.3
    
    pip install requests==2.31.0
  ```