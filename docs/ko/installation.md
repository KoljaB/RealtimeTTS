> **참고:** 이제 `pip install realtimetts`로 기본 설치하는 것은 권장되지 않으며, 대신 `pip install realtimetts[all]`를 사용하세요.

RealtimeTTS 라이브러리는 사용 사례에 맞는 다양한 종속성 설치 옵션을 제공합니다. 여기 당신의 필요에 따라 RealtimeTTS를 설치할 수 있는 다양한 방법이 있습니다:

### 전체 설치

모든 TTS 엔진을 지원하는 RealtimeTTS를 설치하려면:

```plaintext
pip install -U realtimetts[all]
```

### 맞춤 설치

RealtimeTTS는 최소한의 라이브러리 설치로 맞춤 설치를 허용합니다. 다음은 이용 가능한 옵션입니다:
- **모두**: 모든 엔진이 지원되는 전체 설치.
- **시스템**: 시스템별 TTS 기능 포함 (e.g., pyttsx3).
- **azure**: Azure Cognitive Services Speech 지원을 추가합니다.
- **elevenlabs**: ElevenLabs API와의 통합을 포함합니다.
- **openai**: OpenAI 음성 서비스용.
- **gtts**: 구글 텍스트 음성 변환 지원.
- **coqui**: Coqui TTS 엔진을 설치합니다.
- **최소**: 엔진 없이 기본 요구 사항만 설치 (only needed if you want to develop an own engine)


로컬 신경망 Coqui TTS 사용을 위해 RealtimeTTS만 설치하고 싶다고 가정해 보세요, 그러면 다음을 사용해야 합니다:

```plaintext
pip install realtimetts[coqui]
```

예를 들어, Azure Cognitive Services Speech, ElevenLabs, OpenAI 지원만으로 RealtimeTTS를 설치하고 싶다면:

```plaintext
pip install realtimetts[azure,elevenlabs,openai]
```

### 가상 환경 설치

가상 환경 내에서 전체 설치를 수행하려는 분들은 다음 단계를 따르세요:

```
python -m venv env_realtimetts
env_realtimetts\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install -U realtimetts[all]
```

[CUDA 설치](#cuda-installation)에 대한 자세한 정보.

## 엔진 요구 사항

RealtimeTTS에서 지원하는 다양한 엔진은 고유한 요구 사항을 가지고 있습니다. 선택한 엔진에 따라 이러한 요구 사항을 충족하도록 하십시오.

### 시스템엔진
`SystemEngine`은 시스템에 내장된 TTS 기능과 함께 즉시 사용할 수 있습니다. 추가 설정이 필요하지 않습니다.

### GTTSEngine
`GTTSEngine`은 Google Translate의 텍스트 음성 변환 API를 사용하여 즉시 작동합니다. 추가 설정이 필요 없습니다.

### 오픈AI엔진
`OpenAIEngine`을 사용하려면:
- 환경 변수 OPENAI_API_KEY 설정
- ffmpeg 설치 (참고: [CUDA 설치](#cuda-installation) 3번 항목)

### AzureEngine
`AzureEngine`을 사용하려면 다음이 필요합니다:
- Microsoft Azure Text-to-Speech API 키 (AzureEngine 생성자 매개변수 "speech_key" 또는 환경 변수 AZURE_SPEECH_KEY를 통해 제공됨)
- 마이크로소프트 애저 서비스 지역.

`AzureEngine`을 초기화할 때 이러한 자격 증명이 준비되어 있고 올바르게 구성되어 있는지 확인하세요.

### ElevenlabsEngine
`ElevenlabsEngine`을 사용하려면 다음이 필요합니다:
- Elevenlabs API 키 (ElevenlabsEngine 생성자 매개변수 "api_key"를 통해 제공되거나 환경 변수 ELEVENLABS_API_KEY에 설정됨)
- 시스템에 `mpv`가 설치되어 있습니다. (essential for streaming mpeg audio, Elevenlabs only delivers mpeg).

  🔹 **`mpv` 설치하기:**
  - **macOS**:
    ```plaintext
    brew install mpv
    ```

  - **리눅스와 윈도우**: 설치 지침은 [mpv.io](https://mpv.io/)를 방문하세요.

### 코키엔진

고품질의 로컬 신경 TTS를 음성 클로닝과 함께 제공합니다.

먼저 신경 TTS 모델을 다운로드합니다. 대부분의 경우 GPU 합성을 사용하면 실시간으로 충분히 빠를 것입니다. 약 4-5GB의 VRAM이 필요합니다.

- 음성을 복제하려면 소스 음성이 포함된 WAV 파일의 파일 이름을 "voice" 매개변수로 CoquiEngine 생성자에게 제출하십시오.
- 음성 복제는 22050 Hz 모노 16비트 WAV 파일에 짧은 (~5-30초) 샘플이 포함된 경우에 가장 잘 작동합니다.

대부분의 시스템에서는 실시간으로 충분히 빠르게 실행하기 위해 GPU 지원이 필요하며, 그렇지 않으면 끊김 현상이 발생할 것입니다.

### CUDA 설치

이 단계들은 **더 나은 성능**을 요구하고 호환 가능한 NVIDIA GPU를 가진 분들에게 권장됩니다.

> **참고**: *NVIDIA GPU가 CUDA를 지원하는지 확인하려면 [공식 CUDA GPU 목록](https://developer.nvidia.com/cuda-gpus)을 방문하세요.*

CUDA를 통해 지원되는 토치를 사용하려면 다음 단계를 따르세요:

> **참고**: *최신 pytorch 설치는 [여기](https://stackoverflow.com/a/77069523) (확인되지 않음)에서 Toolkit (및 아마도 cuDNN) 설치가 더 이상 필요하지 않을 수 있습니다.*

1. **NVIDIA CUDA 툴킷 설치**:
    예를 들어, Toolkit 12.X를 설치하려면
    - [NVIDIA CUDA 다운로드](https://developer.nvidia.com/cuda-downloads)를 방문하세요.
    - 운영 체제, 시스템 아키텍처 및 OS 버전을 선택하세요.
    - 소프트웨어를 다운로드하고 설치하세요.

    또는 Toolkit 11.8을 설치하려면,
    - [NVIDIA CUDA Toolkit 아카이브](https://developer.nvidia.com/cuda-11-8-0-download-archive)를 방문하세요.
    - 운영 체제, 시스템 아키텍처 및 OS 버전을 선택하세요.
    - 소프트웨어를 다운로드하고 설치하세요.

2. **NVIDIA cuDNN 설치**:

    예를 들어, CUDA 11.x에 cuDNN 8.7.0을 설치하려면
    - [NVIDIA cuDNN 아카이브](https://developer.nvidia.com/rdp/cudnn-archive)를 방문하세요.
    - "Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x"를 클릭하세요.
    - 소프트웨어를 다운로드하고 설치하세요.

3. **ffmpeg 설치**:

    [ffmpeg 웹사이트](https://ffmpeg.org/download.html)에서 운영 체제에 맞는 설치 프로그램을 다운로드할 수 있습니다.

    또는 패키지 관리자를 사용하세요:

    - **우분투나 데비안에서**:
        ```
        sudo apt update && sudo apt install ffmpeg
        ```

    - **Arch Linux에서**:
        ```
        sudo pacman -S ffmpeg
        ```

    - **Homebrew를 사용한 MacOS에서** ([https://brew.sh/](https://brew.sh/)):
        ```bash
brew install ffmpeg
```

    - **Chocolatey를 사용한 Windows에서** ([https://chocolatey.org/](https://chocolatey.org/)):
        ```
        choco install ffmpeg
        ```

    - **Scoop을 사용하여 Windows에서** ([https://scoop.sh/](https://scoop.sh/)):
        ```
        스쿱 설치 ffmpeg
```

4. **CUDA 지원으로 PyTorch 설치하기**:

    CUDA를 사용하여 GPU 지원을 활성화하려면, 특정 CUDA 버전에 따라 다음 지침을 따르세요. CUDA 기능으로 RealtimeSTT의 성능을 향상시키고자 한다면 유용합니다.

    - **CUDA 11.8의 경우:**

        PyTorch와 Torchaudio를 CUDA 11.8을 지원하도록 업데이트하려면 다음 명령어를 사용하세요:

        ```bash
pip install torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```
        텍스트를 번역할 내용: ```

    - **CUDA 12.X의 경우:**


        PyTorch와 Torchaudio를 CUDA 12.X를 지원하도록 업데이트하려면 다음을 실행하세요:

        ```bash
pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
        텍스트를 번역할 내용: ```

    `2.3.1`을(를) 귀하의 시스템 및 요구 사항에 맞는 PyTorch 버전으로 교체하세요.

5. **호환성 문제를 해결하기 위한 수정**:
    라이브러리 호환성 문제에 직면하면, 이 라이브러리들을 고정된 버전으로 설정해 보세요:

  텍스트를 번역할 내용: ``` 

    pip install networkx==2.8.8
    
    pip install typing_extensions==4.8.0
    
    pip install fsspec==2023.6.0
    
    pip install imageio==2.31.6
    
    pip install networkx==2.8.8
    
    pip install numpy==1.24.3
    
    pip install requests==2.31.0
  ```