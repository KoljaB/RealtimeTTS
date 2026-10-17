> **Hinweis:** Die Basisinstallation mit `pip install realtimetts` wird nicht mehr empfohlen. Verwenden Sie stattdessen `pip install realtimetts[all]`.

Die RealtimeTTS-Bibliothek bietet verschiedene Installationsoptionen für Abhängigkeiten je nach Anwendungsfall. Hier sind die verschiedenen Möglichkeiten, RealtimeTTS entsprechend Ihren Anforderungen zu installieren:

### Vollständige Installation

Um RealtimeTTS mit Unterstützung für alle TTS-Engines zu installieren:

```
pip install -U realtimetts[all]
```

### Benutzerdefinierte Installation

RealtimeTTS ermöglicht eine benutzerdefinierte Installation mit minimalen Bibliotheksinstallationen. Folgende Optionen stehen zur Verfügung:
- **all**: Vollständige Installation mit Unterstützung aller Engines.
- **system**: Enthält systemspezifische TTS-Fähigkeiten (z.B. pyttsx3).
- **azure**: Fügt Azure Cognitive Services Speech-Unterstützung hinzu.
- **elevenlabs**: Enthält Integration mit der ElevenLabs API.
- **openai**: Für OpenAI-Sprachdienste.
- **gtts**: Google Text-to-Speech-Unterstützung.
- **coqui**: Installiert die Coqui TTS-Engine.
- **minimal**: Installiert nur die Basisanforderungen ohne Engine (nur erforderlich, wenn Sie eine eigene Engine entwickeln möchten)

Wenn Sie RealtimeTTS nur für die lokale neuronale Coqui TTS-Nutzung installieren möchten, verwenden Sie:

```
pip install realtimetts[coqui]
```

Wenn Sie beispielsweise RealtimeTTS nur mit Azure Cognitive Services Speech, ElevenLabs und OpenAI-Unterstützung installieren möchten:

```
pip install realtimetts[azure,elevenlabs,openai]
```

### Installation in virtueller Umgebung

Für diejenigen, die eine vollständige Installation in einer virtuellen Umgebung durchführen möchten, folgen Sie diesen Schritten:

```
python -m venv env_realtimetts
env_realtimetts\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install -U realtimetts[all]
```

Weitere Informationen zur [CUDA-Installation](#cuda-installation).

## Engine-Anforderungen

Verschiedene von RealtimeTTS unterstützte Engines haben spezifische Anforderungen. Stellen Sie sicher, dass Sie diese Anforderungen je nach gewählter Engine erfüllen.

### SystemEngine
Die `SystemEngine` funktioniert von Haus aus mit den integrierten TTS-Fähigkeiten Ihres Systems. Keine zusätzliche Einrichtung erforderlich.

### GTTSEngine
Die `GTTSEngine` funktioniert von Haus aus mit der Google Translate Text-to-Speech API. Keine zusätzliche Einrichtung erforderlich.

### OpenAIEngine
Zur Verwendung der `OpenAIEngine`:
- Umgebungsvariable OPENAI_API_KEY setzen
- `realtimetts[openai]` installieren (enthält PyAV für die MP3-Wiedergabe).

### AzureEngine
Für die Verwendung der `AzureEngine` benötigen Sie:
- Microsoft Azure Text-to-Speech API-Schlüssel (bereitgestellt über den AzureEngine-Konstruktorparameter "speech_key" oder in der Umgebungsvariable AZURE_SPEECH_KEY)
- Microsoft Azure Service-Region

Stellen Sie sicher, dass diese Anmeldedaten verfügbar und korrekt konfiguriert sind, wenn Sie die `AzureEngine` initialisieren.

### ElevenlabsEngine
Für die `ElevenlabsEngine` benötigen Sie:
- Elevenlabs API-Schlüssel (bereitgestellt über den ElevenlabsEngine-Konstruktorparameter "api_key" oder in der Umgebungsvariable ELEVENLABS_API_KEY)
- `mpv` auf Ihrem System installiert (wesentlich für das Streaming von MPEG-Audio, Elevenlabs liefert nur MPEG)

  🔹 **Installation von `mpv`:**
  - **macOS**:
    ```
    brew install mpv
    ```

  - **Linux und Windows**: Besuchen Sie [mpv.io](https://mpv.io/) für Installationsanweisungen.

### CoquiEngine

Bietet hochwertige, lokale, neuronale TTS mit Stimmklonen.

Lädt zuerst ein neurales TTS-Modell herunter. In den meisten Fällen ist es mit GPU-Synthese schnell genug für Echtzeit. Benötigt etwa 4-5 GB VRAM.

- Um eine Stimme zu klonen, übergeben Sie den Dateinamen einer Wave-Datei, die die Quellstimme enthält, als "voice"-Parameter an den CoquiEngine-Konstruktor
- Stimmklonen funktioniert am besten mit einer 22050 Hz Mono 16bit WAV-Datei, die eine kurze (~5-30 Sek.) Probe enthält

Auf den meisten Systemen wird GPU-Unterstützung benötigt, um schnell genug für Echtzeit zu sein, andernfalls werden Sie Stottern erleben.

### CUDA-Installation

Diese Schritte werden für diejenigen empfohlen, die **bessere Leistung** benötigen und eine kompatible NVIDIA GPU haben.

> **Hinweis**: *Um zu überprüfen, ob Ihre NVIDIA GPU CUDA unterstützt, besuchen Sie die [offizielle CUDA GPUs-Liste](https://developer.nvidia.com/cuda-gpus).*

Um torch mit CUDA-Unterstützung zu verwenden, folgen Sie bitte diesen Schritten:

> **Hinweis**: *Neuere PyTorch-Installationen [könnten](https://stackoverflow.com/a/77069523) (unbestätigt) keine Toolkit (und möglicherweise cuDNN) Installation mehr benötigen.*

1. **NVIDIA CUDA Toolkit installieren**:
    Um beispielsweise Toolkit 12.X zu installieren:
    - Besuchen Sie [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).
    - Wählen Sie Ihr Betriebssystem, Systemarchitektur und OS-Version.
    - Laden Sie die Software herunter und installieren Sie sie.

    oder um Toolkit 11.8 zu installieren:
    - Besuchen Sie [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).
    - Wählen Sie Ihr Betriebssystem, Systemarchitektur und OS-Version.
    - Laden Sie die Software herunter und installieren Sie sie.

2. **NVIDIA cuDNN installieren**:

    Um beispielsweise cuDNN 8.7.0 für CUDA 11.x zu installieren:
    - Besuchen Sie [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
    - Klicken Sie auf "Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x".
    - Laden Sie die Software herunter und installieren Sie sie.

3. **ffmpeg installieren**:

    Sie können einen Installer für Ihr Betriebssystem von der [ffmpeg Website](https://ffmpeg.org/download.html) herunterladen.

    Oder verwenden Sie einen Paketmanager:

    - **Unter Ubuntu oder Debian**:
        ```
        sudo apt update && sudo apt install ffmpeg
        ```

    - **Unter Arch Linux**:
        ```
        sudo pacman -S ffmpeg
        ```

    - **Unter MacOS mit Homebrew** ([https://brew.sh/](https://brew.sh/)):
        ```
        brew install ffmpeg
        ```

    - **Unter Windows mit Chocolatey** ([https://chocolatey.org/](https://chocolatey.org/)):
        ```
        choco install ffmpeg
        ```

    - **Unter Windows mit Scoop** ([https://scoop.sh/](https://scoop.sh/)):
        ```
        scoop install ffmpeg
        ```

4. **PyTorch mit CUDA-Unterstützung installieren**:

    Um Ihre PyTorch-Installation zu aktualisieren und GPU-Unterstützung mit CUDA zu aktivieren, folgen Sie diesen Anweisungen basierend auf Ihrer spezifischen CUDA-Version. Dies ist nützlich, wenn Sie die Leistung von RealtimeSTT mit CUDA-Fähigkeiten verbessern möchten.

    - **Für CUDA 11.8:**

        Um PyTorch und Torchaudio für CUDA 11.8-Unterstützung zu aktualisieren, verwenden Sie folgende Befehle:

        ```
        pip install torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
        ```

    - **Für CUDA 12.X:**

        Um PyTorch und Torchaudio für CUDA 12.X-Unterstützung zu aktualisieren, führen Sie Folgendes aus:

        ```
        pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
        ```

    Ersetzen Sie `2.3.1` durch die Version von PyTorch, die Ihrem System und Ihren Anforderungen entspricht.

5. **Behebung von Kompatibilitätsproblemen**:
    Wenn Sie auf Bibliotheks-Kompatibilitätsprobleme stoßen, versuchen Sie, diese Bibliotheken auf feste Versionen zu setzen:

  `

    pip install networkx==2.8.8
    
    pip install typing_extensions==4.8.0
    
    pip install fsspec==2023.6.0
    
    pip install imageio==2.31.6
    
    pip install networkx==2.8.8
    
    pip install numpy==1.24.3
    
    pip install requests==2.31.0
  `