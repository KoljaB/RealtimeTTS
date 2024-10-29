> **Nota:** L'installazione base con `pip install realtimetts` non è più raccomandata, utilizzare invece `pip install realtimetts[all]`.

La libreria RealtimeTTS fornisce opzioni di installazione per varie dipendenze in base al tuo caso d'uso. Ecco i diversi modi in cui puoi installare RealtimeTTS in base alle tue necessità:

### Installazione Completa

Per installare RealtimeTTS con supporto per tutti i motori TTS:

```
pip install -U realtimetts[all]
```

### Installazione Personalizzata

RealtimeTTS permette un'installazione personalizzata con installazioni minime delle librerie. Ecco le opzioni disponibili:
- **all**: Installazione completa con tutti i motori supportati.
- **system**: Include le capacità TTS specifiche del sistema (es. pyttsx3).
- **azure**: Aggiunge il supporto Azure Cognitive Services Speech.
- **elevenlabs**: Include l'integrazione con l'API ElevenLabs.
- **openai**: Per i servizi vocali OpenAI.
- **gtts**: Supporto Google Text-to-Speech.
- **coqui**: Installa il motore Coqui TTS.
- **minimal**: Installa solo i requisiti base senza motore (necessario solo se si vuole sviluppare un proprio motore)

Se vuoi installare RealtimeTTS solo per l'uso locale di Coqui TTS neurale, dovresti utilizzare:

```
pip install realtimetts[coqui]
```

Per esempio, se vuoi installare RealtimeTTS solo con il supporto per Azure Cognitive Services Speech, ElevenLabs e OpenAI:

```
pip install realtimetts[azure,elevenlabs,openai]
```

### Installazione in Ambiente Virtuale

Per chi vuole eseguire un'installazione completa all'interno di un ambiente virtuale, seguire questi passaggi:

```
python -m venv env_realtimetts
env_realtimetts\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install -U realtimetts[all]
```

Maggiori informazioni sull'[installazione CUDA](#cuda-installation).

## Requisiti dei Motori

I diversi motori supportati da RealtimeTTS hanno requisiti unici. Assicurati di soddisfare questi requisiti in base al motore che scegli.

### SystemEngine
Il `SystemEngine` funziona subito con le capacità TTS integrate nel tuo sistema. Non è necessaria alcuna configurazione aggiuntiva.

### GTTSEngine
Il `GTTSEngine` funziona subito utilizzando l'API text-to-speech di Google Translate. Non è necessaria alcuna configurazione aggiuntiva.

### OpenAIEngine
Per utilizzare `OpenAIEngine`:
- impostare la variabile d'ambiente OPENAI_API_KEY
- installare ffmpeg (vedi [installazione CUDA](#cuda-installation) punto 3)

### AzureEngine
Per utilizzare `AzureEngine`, avrai bisogno di:
- Chiave API Microsoft Azure Text-to-Speech (fornita tramite il parametro del costruttore AzureEngine "speech_key" o nella variabile d'ambiente AZURE_SPEECH_KEY)
- Regione del servizio Microsoft Azure.

Assicurati di avere queste credenziali disponibili e configurate correttamente quando inizializzi `AzureEngine`.

### ElevenlabsEngine
Per `ElevenlabsEngine`, hai bisogno di:
- Chiave API Elevenlabs (fornita tramite il parametro del costruttore ElevenlabsEngine "api_key" o nella variabile d'ambiente ELEVENLABS_API_KEY)
- `mpv` installato sul tuo sistema (essenziale per lo streaming audio mpeg, Elevenlabs fornisce solo mpeg).

  🔹 **Installazione di `mpv`:**
  - **macOS**:
    ```
    brew install mpv
    ```

  - **Linux e Windows**: Visita [mpv.io](https://mpv.io/) per le istruzioni di installazione.

### CoquiEngine

Fornisce TTS neurale locale di alta qualità con clonazione vocale.

Scarica prima un modello TTS neurale. Nella maggior parte dei casi sarà abbastanza veloce per il tempo reale utilizzando la sintesi GPU. Richiede circa 4-5 GB di VRAM.

- per clonare una voce inviare il nome del file di un file wave contenente la voce sorgente come parametro "voice" al costruttore CoquiEngine
- la clonazione vocale funziona meglio con un file WAV mono 16bit a 22050 Hz contenente un breve campione (circa 5-30 sec)

Sulla maggior parte dei sistemi sarà necessario il supporto GPU per funzionare abbastanza velocemente per il tempo reale, altrimenti si verificheranno interruzioni.

### Installazione CUDA

Questi passaggi sono raccomandati per chi richiede **migliori prestazioni** e ha una GPU NVIDIA compatibile.

> **Nota**: *per verificare se la tua GPU NVIDIA supporta CUDA, visita la [lista ufficiale delle GPU CUDA](https://developer.nvidia.com/cuda-gpus).*

Per utilizzare torch con supporto via CUDA segui questi passaggi:

> **Nota**: *le installazioni più recenti di pytorch [potrebbero](https://stackoverflow.com/a/77069523) (non verificato) non necessitare più dell'installazione del Toolkit (e possibilmente cuDNN).*

1. **Installa NVIDIA CUDA Toolkit**:
    Per esempio, per installare il Toolkit 12.X, per favore
    - Visita [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).
    - Seleziona il tuo sistema operativo, architettura di sistema e versione del sistema operativo.
    - Scarica e installa il software.

    o per installare il Toolkit 11.8, per favore
    - Visita [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).
    - Seleziona il tuo sistema operativo, architettura di sistema e versione del sistema operativo.
    - Scarica e installa il software.

2. **Installa NVIDIA cuDNN**:

    Per esempio, per installare cuDNN 8.7.0 per CUDA 11.x per favore
    - Visita [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
    - Clicca su "Download cuDNN v8.7.0 (28 Novembre 2022), per CUDA 11.x".
    - Scarica e installa il software.

3. **Installa ffmpeg**:

    Puoi scaricare un installer per il tuo sistema operativo dal [sito web ffmpeg](https://ffmpeg.org/download.html).

    Oppure usa un gestore pacchetti:

    - **Su Ubuntu o Debian**:
        ```
        sudo apt update && sudo apt install ffmpeg
        ```

    - **Su Arch Linux**:
        ```
        sudo pacman -S ffmpeg
        ```

    - **Su MacOS usando Homebrew** ([https://brew.sh/](https://brew.sh/)):
        ```
        brew install ffmpeg
        ```

    - **Su Windows usando Chocolatey** ([https://chocolatey.org/](https://chocolatey.org/)):
        ```
        choco install ffmpeg
        ```

    - **Su Windows usando Scoop** ([https://scoop.sh/](https://scoop.sh/)):
        ```
        scoop install ffmpeg
        ```

4. **Installa PyTorch con supporto CUDA**:

    Per aggiornare la tua installazione PyTorch per abilitare il supporto GPU con CUDA, segui queste istruzioni basate sulla tua versione CUDA specifica. Questo è utile se desideri migliorare le prestazioni di RealtimeSTT con le capacità CUDA.

    - **Per CUDA 11.8:**

        Per aggiornare PyTorch e Torchaudio per supportare CUDA 11.8, usa i seguenti comandi:

        ```
        pip install torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
        ```

    - **Per CUDA 12.X:**

        Per aggiornare PyTorch e Torchaudio per supportare CUDA 12.X, esegui quanto segue:

        ```
        pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
        ```

    Sostituisci `2.3.1` con la versione di PyTorch che corrisponde al tuo sistema e requisiti.

5. **Fix per risolvere problemi di compatibilità**:
    Se riscontri problemi di compatibilità delle librerie, prova a impostare queste librerie a versioni fisse:

  ```
  
    pip install networkx==2.8.8
    
    pip install typing_extensions==4.8.0
    
    pip install fsspec==2023.6.0
    
    pip install imageio==2.31.6
    
    pip install networkx==2.8.8
    
    pip install numpy==1.24.3
    
    pip install requests==2.31.0
  ```