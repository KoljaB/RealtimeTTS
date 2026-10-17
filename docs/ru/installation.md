> **Примечание:** Базовая установка с помощью `pip install realtimetts` больше не рекомендуется, вместо этого используйте `pip install realtimetts[all]`.

Библиотека RealtimeTTS предоставляет варианты установки различных зависимостей в зависимости от вашего случая использования. Вот различные способы установки RealtimeTTS в зависимости от ваших потребностей:

### Полная установка

Чтобы установить RealtimeTTS с поддержкой всех TTS-движков:

```
pip install -U realtimetts[all]
```

### Пользовательская установка

RealtimeTTS позволяет выполнять индивидуальную установку с минимальным количеством библиотек. Вот доступные варианты:
- **все**: Полная установка со всеми поддерживаемыми движками.
- **система**: Включает возможности TTS, специфичные для системы (e.g., pyttsx3).
- **azure**: Добавляет поддержку Azure Cognitive Services Speech.
- **elevenlabs**: Включает интеграцию с API ElevenLabs.
- **openai**: Для голосовых сервисов OpenAI.
- **gtts**: Поддержка Google Text-to-Speech.
- **coqui**: Устанавливает движок Coqui TTS.
- **minimal**: Устанавливает только базовые требования без движка (only needed if you want to develop an own engine)


Скажем, вы хотите установить RealtimeTTS только для локального использования нейронного Coqui TTS, тогда вам следует использовать:

```
pip install realtimetts[coqui]
```

Например, если вы хотите установить RealtimeTTS только с поддержкой Azure Cognitive Services Speech, ElevenLabs и OpenAI:

```
pip install realtimetts[azure,elevenlabs,openai]
```

### Установка в виртуальной среде

Для тех, кто хочет выполнить полную установку в виртуальной среде, выполните следующие шаги:

```
python -m venv env_realtimetts
env_realtimetts\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install -U realtimetts[all]
```

Больше информации о [установке CUDA](#cuda-installation).

## Требования к движкам

Разные движки, поддерживаемые RealtimeTTS, имеют уникальные требования. Убедитесь, что вы выполняете эти требования в зависимости от выбранного вами движка.

### SystemEngine
`SystemEngine` работает сразу после установки с встроенными возможностями TTS вашей системы. Дополнительная настройка не требуется.

### GTTSEngine
`GTTSEngine` работает из коробки, используя API синтеза речи Google Translate. Дополнительная настройка не требуется.

### OpenAIEngine
Чтобы использовать `OpenAIEngine`:
- установите переменную окружения OPENAI_API_KEY
- установите `realtimetts[openai]` (включает PyAV для воспроизведения MP3).

### AzureEngine
Чтобы использовать `AzureEngine`, вам потребуется:
- Ключ API Microsoft Azure Text-to-Speech (предоставляется через параметр конструктора AzureEngine "speech_key" или в переменной окружения AZURE_SPEECH_KEY)
- Регион службы Microsoft Azure.

Убедитесь, что у вас есть эти учетные данные и они правильно настроены при инициализации `AzureEngine`.

### ElevenlabsEngine
Для `ElevenlabsEngine` вам нужно:
- Ключ API Elevenlabs (предоставляется через параметр конструктора ElevenlabsEngine "api_key" или в переменной окружения ELEVENLABS_API_KEY)
- `mpv` установлен на вашей системе (essential for streaming mpeg audio, Elevenlabs only delivers mpeg).

  🔹 **Установка `mpv`:**
  - **macOS**:
    ```
    brew install mpv
    ```

  - **Linux и Windows**: Посетите [mpv.io](https://mpv.io/) для получения инструкций по установке.

### CoquiEngine

Предоставляет высококачественный, локальный, нейронный TTS с клонированием голоса.

Сначала загружает нейронную модель TTS. В большинстве случаев это будет достаточно быстро для синтеза в реальном времени с использованием GPU. Нужен около 4-5 ГБ видеопамяти.

- чтобы клонировать голос, укажите имя файла WAV, содержащего исходный голос, в качестве параметра "voice" в конструкторе CoquiEngine
- клонирование голоса работает лучше всего с монофоническим WAV-файлом 22050 Гц 16 бит, содержащим короткий (~5-30 сек) образец

На большинстве систем потребуется поддержка GPU, чтобы работать достаточно быстро для реального времени, иначе вы будете испытывать заикания.

### Установка CUDA

Эти шаги рекомендуются тем, кто требует **лучшей производительности** и имеет совместимую видеокарту NVIDIA.

> **Примечание**: *чтобы проверить, поддерживает ли ваш графический процессор NVIDIA CUDA, посетите [официальный список графических процессоров CUDA](https://developer.nvidia.com/cuda-gpus).*

Чтобы использовать torch с поддержкой через CUDA, пожалуйста, выполните следующие шаги:

> **Примечание**: *новые установки pytorch [возможно](https://stackoverflow.com/a/77069523) (неподтверждено) больше не требуют установки Toolkit (и, возможно, cuDNN).*

1. **Установите NVIDIA CUDA Toolkit**:
    Например, чтобы установить Toolkit 12.X, пожалуйста
    - Посетите [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).
    - Выберите вашу операционную систему, архитектуру системы и версию ОС.
    - Скачайте и установите программное обеспечение.

    или для установки Toolkit 11.8, пожалуйста
    - Посетите [архив NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive).
    - Выберите вашу операционную систему, архитектуру системы и версию ОС.
    - Скачайте и установите программное обеспечение.

2. **Установите NVIDIA cuDNN**:

    Например, чтобы установить cuDNN 8.7.0 для CUDA 11.x, пожалуйста,
    - Посетите [архив NVIDIA cuDNN](https://developer.nvidia.com/rdp/cudnn-archive).
    - Нажмите на "Скачать cuDNN v8.7.0 (28 ноября 2022 года), для CUDA 11.x".
    - Скачайте и установите программное обеспечение.

3. **Установите ffmpeg**:

    Вы можете скачать установщик для вашей ОС с [веб-сайта ffmpeg](https://ffmpeg.org/download.html).

    Или используйте менеджер пакетов:

    - **На Ubuntu или Debian**:
        ```
        sudo apt update && sudo apt install ffmpeg
        ```

    - **На Arch Linux**:
        ```
        sudo pacman -S ffmpeg
        ```

    - **На MacOS с использованием Homebrew** ([https://brew.sh/](https://brew.sh/)):
        ```
        brew install ffmpeg
        ```

    - **На Windows с использованием Chocolatey** ([https://chocolatey.org/](https://chocolatey.org/)):
        ```
        choco install ffmpeg
        ```

    - **На Windows с использованием Scoop** ([https://scoop.sh/](https://scoop.sh/)):
        ```
        scoop install ffmpeg
        ```

4. **Установите PyTorch с поддержкой CUDA**:

    Чтобы обновить вашу установку PyTorch для включения поддержки GPU с CUDA, следуйте этим инструкциям в зависимости от вашей конкретной версии CUDA. Это полезно, если вы хотите улучшить производительность RealtimeSTT с помощью возможностей CUDA.

    - **Для CUDA 11.8:**

        Чтобы обновить PyTorch и Torchaudio для поддержки CUDA 11.8, используйте следующие команды:

        ```
        pip install torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```
        ```

    - **Для CUDA 12.X:**


        Чтобы обновить PyTorch и Torchaudio для поддержки CUDA 12.X, выполните следующее:

        ```
        pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
        ```

    Замените `2.3.1` на версию PyTorch, которая соответствует вашей системе и требованиям.

5. **Исправление для решения проблем совместимости**:
    Если вы столкнетесь с проблемами совместимости библиотек, попробуйте установить эти библиотеки на фиксированные версии:

  ``` 

    pip install networkx==2.8.8
    
    pip install typing_extensions==4.8.0
    
    pip install fsspec==2023.6.0
    
    pip install imageio==2.31.6
    
    pip install networkx==2.8.8
    
    pip install numpy==1.24.3
    
    pip install requests==2.31.0
  ```