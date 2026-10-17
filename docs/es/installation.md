> **Nota:** Ya no se recomienda la instalación básica con `pip install realtimetts`, use `pip install realtimetts[all]` en su lugar.

La biblioteca RealtimeTTS proporciona opciones de instalación para varias dependencias según su caso de uso. Aquí están las diferentes formas en que puede instalar RealtimeTTS según sus necesidades:

### Instalación Completa

Para instalar RealtimeTTS con soporte para todos los motores de TTS:

```
pip install -U realtimetts[all]
```

### Instalación Personalizada

RealtimeTTS permite una instalación personalizada con instalaciones mínimas de bibliotecas. Estas son las opciones disponibles:
- **all**: Instalación completa con todos los motores soportados.
- **system**: Incluye capacidades de TTS específicas del sistema (por ejemplo, pyttsx3).
- **azure**: Agrega soporte para Azure Cognitive Services Speech.
- **elevenlabs**: Incluye integración con la API de ElevenLabs.
- **openai**: Para servicios de voz de OpenAI.
- **gtts**: Soporte para Google Text-to-Speech.
- **coqui**: Instala el motor Coqui TTS.
- **minimal**: Instala solo los requisitos base sin motor (solo necesario si desea desarrollar un motor propio)

Por ejemplo, si desea instalar RealtimeTTS solo para uso local de Coqui TTS neuronal, debe usar:

```
pip install realtimetts[coqui]
```

Si desea instalar RealtimeTTS solo con Azure Cognitive Services Speech, ElevenLabs y soporte de OpenAI:

```
pip install realtimetts[azure,elevenlabs,openai]
```

### Instalación en Entorno Virtual

Para aquellos que deseen realizar una instalación completa dentro de un entorno virtual, sigan estos pasos:

```
python -m venv env_realtimetts
env_realtimetts\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install -U realtimetts[all]
```

Más información sobre [instalación de CUDA](#instalación-de-cuda).

## Requisitos de los Motores

Los diferentes motores soportados por RealtimeTTS tienen requisitos únicos. Asegúrese de cumplir con estos requisitos según el motor que elija.

### SystemEngine
El `SystemEngine` funciona de inmediato con las capacidades de TTS incorporadas en su sistema. No se necesita configuración adicional.

### GTTSEngine
El `GTTSEngine` funciona de inmediato usando la API de texto a voz de Google Translate. No se necesita configuración adicional.

### OpenAIEngine
Para usar el `OpenAIEngine`:
- configure la variable de entorno OPENAI_API_KEY
- instale `realtimetts[openai]` (incluye PyAV para reproducir MP3).

### AzureEngine
Para usar el `AzureEngine`, necesitará:
- Clave API de Microsoft Azure Text-to-Speech (proporcionada a través del parámetro "speech_key" del constructor AzureEngine o en la variable de entorno AZURE_SPEECH_KEY)
- Región de servicio de Microsoft Azure.

Asegúrese de tener estas credenciales disponibles y correctamente configuradas al inicializar el `AzureEngine`.

### ElevenlabsEngine
Para el `ElevenlabsEngine`, necesita:
- Clave API de Elevenlabs (proporcionada a través del parámetro "api_key" del constructor ElevenlabsEngine o en la variable de entorno ELEVENLABS_API_KEY)
- `mpv` instalado en su sistema (esencial para transmitir audio mpeg, Elevenlabs solo entrega mpeg).

  🔹 **Instalación de `mpv`:**
  - **macOS**:
    ```
    brew install mpv
    ```

  - **Linux y Windows**: Visite [mpv.io](https://mpv.io/) para instrucciones de instalación.

### CoquiEngine

Proporciona TTS neuronal local de alta calidad con clonación de voz.

Descarga primero un modelo neuronal TTS. En la mayoría de los casos, será lo suficientemente rápido para tiempo real usando síntesis GPU. Necesita alrededor de 4-5 GB de VRAM.

- para clonar una voz, envíe el nombre del archivo de un archivo wave que contenga la voz fuente como parámetro "voice" al constructor CoquiEngine
- la clonación de voz funciona mejor con un archivo WAV mono de 16 bits a 22050 Hz que contenga una muestra corta (~5-30 seg)

En la mayoría de los sistemas, se necesitará soporte de GPU para ejecutarse lo suficientemente rápido en tiempo real, de lo contrario experimentará tartamudeo.

### Instalación de CUDA

Estos pasos son recomendados para aquellos que requieren **mejor rendimiento** y tienen una GPU NVIDIA compatible.

> **Nota**: *para verificar si su GPU NVIDIA es compatible con CUDA, visite la [lista oficial de GPUs CUDA](https://developer.nvidia.com/cuda-gpus).*

Para usar torch con soporte vía CUDA, siga estos pasos:

> **Nota**: *las instalaciones más nuevas de pytorch [pueden](https://stackoverflow.com/a/77069523) (no verificado) no necesitar la instalación de Toolkit (y posiblemente cuDNN).*

1. **Instalar NVIDIA CUDA Toolkit**:
    Por ejemplo, para instalar Toolkit 12.X, por favor
    - Visite [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).
    - Seleccione su sistema operativo, arquitectura del sistema y versión del sistema operativo.
    - Descargue e instale el software.

    o para instalar Toolkit 11.8, por favor
    - Visite [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).
    - Seleccione su sistema operativo, arquitectura del sistema y versión del sistema operativo.
    - Descargue e instale el software.

2. **Instalar NVIDIA cuDNN**:

    Por ejemplo, para instalar cuDNN 8.7.0 para CUDA 11.x por favor
    - Visite [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
    - Haga clic en "Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x".
    - Descargue e instale el software.

3. **Instalar ffmpeg**:

    Puede descargar un instalador para su sistema operativo desde el [sitio web de ffmpeg](https://ffmpeg.org/download.html).

    O usar un gestor de paquetes:

    - **En Ubuntu o Debian**:
        ```
        sudo apt update && sudo apt install ffmpeg
        ```

    - **En Arch Linux**:
        ```
        sudo pacman -S ffmpeg
        ```

    - **En MacOS usando Homebrew** ([https://brew.sh/](https://brew.sh/)):
        ```
        brew install ffmpeg
        ```

    - **En Windows usando Chocolatey** ([https://chocolatey.org/](https://chocolatey.org/)):
        ```
        choco install ffmpeg
        ```

    - **En Windows usando Scoop** ([https://scoop.sh/](https://scoop.sh/)):
        ```
        scoop install ffmpeg
        ```

4. **Instalar PyTorch con soporte CUDA**:

    Para actualizar su instalación de PyTorch y habilitar el soporte de GPU con CUDA, siga estas instrucciones según su versión específica de CUDA. Esto es útil si desea mejorar el rendimiento de RealtimeSTT con capacidades CUDA.

    - **Para CUDA 11.8:**

        Para actualizar PyTorch y Torchaudio para soportar CUDA 11.8, use los siguientes comandos:

        ```
        pip install torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
        ```

    - **Para CUDA 12.X:**

        Para actualizar PyTorch y Torchaudio para soportar CUDA 12.X, ejecute lo siguiente:

        ```
        pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
        ```

    Reemplace `2.3.1` con la versión de PyTorch que coincida con su sistema y requisitos.

5. **Solución para resolver problemas de compatibilidad**:
    Si encuentra problemas de compatibilidad de bibliotecas, intente establecer estas bibliotecas en versiones fijas:

    ```
    pip install networkx==2.8.8
    pip install typing_extensions==4.8.0
    pip install fsspec==2023.6.0
    pip install imageio==2.31.6
    pip install networkx==2.8.8
    pip install numpy==1.24.3
    pip install requests==2.31.0
    ```