
> **Remarque:** Installation de base avec `pip install realtimetts`s n'est plus recommandé, utilisez `pip install realtimetts[all]` à la place.

La bibliothèque RealtimeTTS offre des options d'installation pour diverses dépendances pour votre cas d'utilisation. Voici les différentes façons dont vous pouvez installer RealtimeTTS en fonction de vos besoins :

### Installation complète

Pour installer RealtimeTTS avec prise en charge de tous les moteurs TTS :

``
pip install -U realtimetts [tous]
``

### Installation personnalisée

RealtimeTTS permet une installation personnalisée avec un minimum d'installations de bibliothèque. Voici les options disponibles :
- **all** : Installation complète avec chaque moteur pris en charge.
- ** système** : Inclut les capacités TTS spécifiques au système (par exemple, pyttsx3).
- **azure** : ajoute le support vocal Azure Cognitive Services.
- **elevenlabs** : Comprend l'intégration avec l'API ElevenLabs.
- **openai** : Pour les services vocaux OpenAI.
- **gtts** : Prise en charge de Google Text-to-Speech.
- **coqui** : Installe le moteur Coqui TTS.
- **minimal** : installe uniquement les exigences de base sans moteur (nécessaire uniquement si vous souhaitez développer votre propre moteur)


Supposons que vous souhaitiez installer RealtimeTTS uniquement pour l'utilisation neuronale locale de Coqui TTS, vous devez alors utiliser :

``
pip installez realtimetts [coqui]
``

Par exemple, si vous souhaitez installer RealtimeTTS avec uniquement Azure Cognitive Services Speech, ElevenLabs et la prise en charge d'OpenAI :

``
pip installez realtimetts[azure,elevenlabs,openai]
``

### Installation de l'environnement virtuel

Pour ceux qui souhaitent effectuer une installation complète dans un environnement virtuel, procédez comme suit

``
python - m venv env_realtimetts
env_realtimetts\Scripts\activate.bat
python.exe - m pip install - upgrade pip
pip install -U realtimetts [tous]
``

Plus d'informations sur installation CUDA.

## Exigences du moteur

Différents moteurs pris en charge par RealtimeTTS ont des exigences uniques. Assurez-vous de remplir ces exigences en fonction du moteur que vous choisissez.

### Moteur système
Le `SystemEngine fonctionne dès le départ avec les capacités TTS intégrées de votre système. Aucune configuration supplémentaire n'est nécessaire.

### GTTSEngine
Le `GTTSEngine` fonctionne dès le départ à l'aide de l'API de synthèse vocale de Google Translate. Aucune configuration supplémentaire n'est nécessaire.

### OpenAIEngine
Pour utiliser le ``(OpenAIE):
- définir la variable d'environnement OPENAI_API_KEY
- installer `realtimetts[openai]` (inclut PyAV pour la lecture MP3).

### AzureEngine
Pour utiliser le `ine`, vous aurez besoin de :
- Clé API Microsoft Azure Text-to-Speech (fournie via le paramètre constructeur AzureEngine « speech_key » ou dans la variable d'environnement AZURE_SPEECH_KEY)
- Région de service Microsoft Azure.

Assurez-vous d'avoir ces informations d'identification disponibles et correctement configurées lors de l'initialisation du `AzureEngine`.

### ElevenlabsEngine
Pour le `ElevenlabsEngine`, vous avez besoin de:
- Clé API Elevenlabs (fournie via le paramètre constructeur ElevenlabsEngine « api_key » ou dans la variable d'environnement ELEVENLABS_API_KEY)
- `mpv` installed on your system (essential for streaming mpeg audio, Elevenlabs ne délivre que mpeg).

### ElevenlabsEngine
Pour le `ElevenlabsEngine`, vous avez besoin de:
- Clé API Elevenlabs (fournie via le paramètre constructeur ElevenlabsEngine « api_key » ou dans la variable d'environnement ELEVENLABS_API_KEY)
- `mpv` installed on your system (essential for streaming mpeg audio, Elevenlabs ne délivre que mpeg).

  🔹 **Installation `v`:**
  - **macOS**:
    ``
    infuser installer mpv
    ``

  - **Linux et Windows** : Visitez [mpv.io](https://mpv.io/) pour les instructions d'installation.

### CoquiEngine

Offre un TTS neuronal local de haute qualité avec clonage vocal.

Télécharge d'abord un modèle TTS neuronal. Dans la plupart des cas, il est suffisamment rapide pour le temps réel utilisant la synthèse GPU. Nécessite environ 4 à 5 Go de VRAM.

- pour cloner une voix, soumettez le nom de fichier d'un fichier d'onde contenant la voix source comme paramètre « voix » au constructeur CoquiEngine
- le clonage vocal fonctionne mieux avec un fichier WAV mono 16 bits de 22 050 Hz contenant un échantillon court (~5 à 30 secondes)

Sur la plupart des systèmes, la prise en charge du GPU sera nécessaire pour fonctionner suffisamment rapidement en temps réel, sinon vous ferez l'expérience du bégaiement.

### Installation CUDA

Ces étapes sont recommandées pour ceux qui ont besoin de ** meilleures performances ** et disposent d'un GPU NVIDIA compatible.

> **Remarque** : *pour vérifier si votre GPU NVIDIA prend en charge CUDA, visitez la [liste officielle des GPU CUDA](https://developer.nvidia.com/cuda-gpus).*

Pour utiliser une torche avec support via CUDA, veuillez suivre ces étapes :

> **Remarque** : *les installations de pythorque plus récentes [peuvent](https://stackoverflow.com/a/77069523) (non vérifié) n'ont plus besoin d'installation de Toolkit (et éventuellement de cuDNN).*

1. **Installer NVIDIA CUDA Toolkit**:
    Par exemple, pour installer Toolkit 12.X, s'il te plaît
    - Visitez [NVIDIA CUDA Téléchargements](https://developer.nvidia.com/cuda-downloads).
    - Sélectionnez votre système d'exploitation, votre architecture système et votre version os.
    - Téléchargez et installez le logiciel.

    ou pour installer Toolkit 11.8, s'il vous plaît
    - Visitez [Archive de la boîte à outils CUDA NVIDIA](https://developer.nvidia.com/cuda-11-8-0-download-archive).
    - Sélectionnez votre système d'exploitation, votre architecture système et votre version os.
    - Téléchargez et installez le logiciel.

2. **Installer NVIDIA cuDNN**:

    Par exemple, pour installer cuDNN 8.7.0 pour CUDA 11. x s'il vous plaît
    - Visitez [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
    - Cliquez sur « Télécharger cuDNN v8.7.0 (28 novembre 2022), pour CUDA 11.x ».
    - Téléchargez et installez le logiciel.

3. **Installer ffmpeg**:

    Vous pouvez télécharger un programme d'installation pour votre système d'exploitation à partir du [site Web deffmpeg](https://ffmpeg.org/download.html).

    Ou utilisez un gestionnaire de packages :

    - **Sur Ubuntu ou Debian**:
        ``
        sudo apt update & & sudo apt install ffmpeg
        ``

    - **Sur Arch Linux**:
        ``
        sudo pacman -S ffmpeg
        ``

    - **Sur MacOS utilisant Homebrew** ([https://brew.sh/](https://brew.sh/)):
        ``
        infuser installer ffmpeg
        ``

    - **Sur Windows utilisant Chocolatey** ([https://chocolatey.org/](https://chocolatey.org/)):
        ``
        choco installer ffmpeg
        ``

    - **Sur Windows utilisant Scoop** ([https://scoop.sh/](https://scoop.sh/)):
        ``
        scoop installer ffmpeg
        ``

4. **Installez PyTorch avec le support CUDA** :

    Pour mettre à niveau votre installation PyTorch afin d'activer le support GPU avec CUDA, suivez ces instructions en fonction de votre version CUDA spécifique. Ceci est utile si vous souhaitez améliorer les performances de RealtimeSTT avec les capacités CUDA.

    - **Pour CUDA 11.8:**

        Pour mettre à jour PyTorch et Torchaudio afin de prendre en charge CUDA 11.8, utilisez les commandes suivantes :

        ``
        pip installe torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
        ``

    - **Pour CUDA 12.X:**


        Pour mettre à jour PyTorch et Torchaudio pour prendre en charge CUDA 12.X, exécutez ce qui suit :

        ``
        pip installe torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
        ``

    Remplacer ` ` ` of PyTorch that matching your system and requirements.

5. ** Correction pour résoudre les problèmes de compatibilité** :
    Si vous rencontrez des problèmes de compatibilité de bibliothèque, essayez de définir ces bibliothèques sur des versions fixes :

   ``` 

    pip install networkx==2.8.8
    
    pip install typing_extensions==4.8.0
    
    pip install fsspec==2023.6.0
    
    pip install imageio==2.31.6
    
    pip install networkx==2.8.8
    
    pip install numpy==1.24.3
    
    pip install requests==2.31.0
  ```