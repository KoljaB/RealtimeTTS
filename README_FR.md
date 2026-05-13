# RealtimeTTS
[![PyPI](https://img.shields.io/pypi/v/RealtimeTTS)](https://pypi.org/project/RealtimeTTS/)
[![Téléchargements](https://static.pepy.tech/badge/RealtimeTTS)](https://pepy.tech/project/KoljaB/RealtimeTTS)
[![Version GitHub](https://img.shields.io/github/release/KoljaB/RealtimeTTS.svg)](https://GitHub.com/KoljaB/RealtimeTTS/releases/)
[![GitHub commits](https://badgen.net/github/commits/KoljaB/RealtimeTTS)](https://GitHub.com/Naereen/KoljaB/RealtimeTTS/commit/)
[![GitHub forks](https://img.shields.io/github/forks/KoljaB/RealtimeTTS.svg?style=social&label=Fork&maxAge=2592000)](https://GitHub.com/KoljaB/RealtimeTTS/network/)
[![GitHub étoiles](https://img.shields.io/github/stars/KoljaB/RealtimeTTS.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/KoljaB/RealtimeTTS/stargazers/)

*Bibliothèque de synthèse vocale à faible latence et facile à utiliser pour les applications en temps réel*

## À propos du projet

RealtimeTTS est une bibliothèque de synthèse vocale (TTS) de pointe conçue pour les applications en temps réel. Elle se distingue par sa capacité à convertir rapidement des flux de texte en sortie auditive de haute qualité avec une latence minimale.

> **Important :** [Installation](#installation) a changé pour permettre plus de personnalisation. Veuillez utiliser `pip install realtimetts[all]` au lieu de `pip install realtimetts` maintenant. Plus d'informations ici](#installation).

> **Astuce :** *<strong>Consultez [Linguflex](https://github.com/KoljaB/Linguflex)</strong>, le projet original dont RealtimeTTS est issu. Il vous permet de contrôler votre environnement en parlant et est l'un des assistants open source les plus performants et les plus sophistiqués actuellement disponibles.*

https://github.com/KoljaB/RealtimeTTS/assets/7604638/87dcd9a5-3a4e-4f57-be45-837fc63237e7

## Principales fonctionnalités

- **Faible latence**
- conversion texte-parole quasi instantanée
- compatible avec les sorties LLM
- **Audio de haute qualité**
- génère une parole claire et naturelle
- **Prise en charge de plusieurs moteurs TTS**
- prend en charge OpenAI TTS, Elevenlabs, Azure Speech Services, Coqui TTS, gTTS et System TTS
- **Multilingue**
- **Robuste et fiable** :
- assure un fonctionnement continu grâce à un mécanisme de secours
- bascule vers des moteurs alternatifs en cas de perturbations garantissant des performances et une fiabilité constantes, ce qui est essentiel pour une utilisation critique et professionnelle cas

> **Astuce** : *consultez [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT), l'équivalent d'entrée de cette bibliothèque, pour les capacités de conversion de la parole en texte. Ensemble, ils forment un puissant wrapper audio en temps réel autour de grands modèles de langage.*

## FAQ

Consultez la [page FAQ](./FAQ.md) pour obtenir des réponses à de nombreuses questions sur l'utilisation de RealtimeTTS.

## Mises à jour

Dernière version : v0.4.7

Voir [historique des versions](https://github.com/KoljaB/RealtimeTTS/releases).

## Pile technologique

Cette bibliothèque utilise :

- **Moteurs de synthèse vocale**
- **OpenAIEngine** : le système TTS d'OpenAI offre 6 voix au son naturel.
- **CoquiEngine** : TTS neuronal local de haute qualité.
- **AzureEngine** : la technologie TTS de pointe de Microsoft. 500 000 caractères gratuits par mois.
- **ElevenlabsEngine** : offre les meilleures voix disponibles.
- **GTTSEngine** : utilisation gratuite et ne nécessite pas la configuration d'un GPU local.
- **SystemEngine** : moteur natif pour une configuration rapide.

- **Sentence Boundary Detection**
- **NLTK Sentence Tokenizer** : tokenizer de phrases de Natural Language Toolkit pour des tâches de synthèse vocale simples en anglais ou lorsque la simplicité est préférée.
- **Stanza Sentence Tokenizer** : tokenizer de phrases Stanza pour travailler avec du texte multilingue ou lorsqu'une précision et des performances supérieures sont requises.

*En utilisant des composants « standards de l'industrie », RealtimeTTS offre une base technologique fiable et haut de gamme pour le développement de solutions vocales avancées.*

## Installation

> **Remarque :** l'installation de base avec `pip install realtimetts` n'est plus recommandée, utilisez plutôt `pip install realtimetts[all]`.

La bibliothèque RealtimeTTS fournit des options d'installation pour diverses dépendances pour votre cas d'utilisation. Voici les différentes manières d'installer RealtimeTTS en fonction de vos besoins :

### Installation complète

Pour installer RealtimeTTS avec prise en charge de tous les moteurs TTS :

```bash
pip install -U realtimetts[all]
```

### Installation personnalisée

RealtimeTTS permet une installation personnalisée avec des installations de bibliothèque minimales. Voici les options disponibles :
- **all** : installation complète avec prise en charge de tous les moteurs.
- **system** : inclut des fonctionnalités TTS spécifiques au système (par exemple, pyttsx3).
- **azure** : ajoute la prise en charge de la parole par Azure Cognitive Services.
- **elevenlabs** : inclut l'intégration avec l'API ElevenLabs.
- **openai** : pour les services vocaux OpenAI.
- **gtts** : prise en charge de la synthèse vocale par Google.
- **coqui** : installe le moteur TTS Coqui.
- **minimal** : installe uniquement les exigences de base sans moteur (uniquement nécessaire si vous souhaitez développer votre propre moteur)

Supposons que vous souhaitiez installer RealtimeTTS uniquement pour l'utilisation locale de Coqui TTS neuronal, vous devez alors utiliser :

```bash
pip install realtimetts[coqui]
```

Par exemple, si vous souhaitez installer Real timeTTS avec prise en charge d'Azure Cognitive Services Speech, ElevenLabs et OpenAI uniquement :

```bash
pip install realtimetts[azure,elevenlabs,openai]
```

### Installation d'un environnement virtuel

Pour ceux qui souhaitent effectuer une installation complète dans un environnement virtuel, suivez ces étapes :

```bash
python -m venv env_realtimetts
env_realtimetts\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install -U realtimetts[all]
```

Plus d'informations sur [l'installation CUDA](#cuda-installation).

## Configuration requise pour le moteur

Les différents moteurs pris en charge par RealtimeTTS ont des exigences uniques. Assurez-vous de répondre à ces exigences en fonction du moteur que vous choisissez.

### SystemEngine
`SystemEngine` fonctionne immédiatement avec les fonctionnalités TTS intégrées de votre système. Aucune configuration supplémentaire n'est nécessaire.

### GTTSEngine
Le `GTTSEngine` fonctionne immédiatement à l'aide de l'API de synthèse vocale de Google Translate. Aucune configuration supplémentaire n'est nécessaire.

### OpenAIEngine
Pour utiliser `OpenAIEngine` :
- définissez la variable d'environnement OPENAI_API_KEY
- installez ffmpeg (voir [Installation CUDA](#cuda-installation) point 3)

### AzureEngine
Pour utiliser `AzureEngine`, vous aurez besoin de :
- Clé API de synthèse vocale Microsoft Azure (fournie via le paramètre de constructeur AzureEngine « speech_key » ou dans la variable d'environnement AZURE_SPEECH_KEY)
- Région de service Microsoft Azure.

Assurez-vous que ces informations d'identification sont disponibles et correctement configurées lors de l'initialisation de `AzureEngine`.

### ElevenlabsEngine
Pour `ElevenlabsEngine`, vous avez besoin de :
- Clé API Elevenlabs (fournie via le paramètre constructeur ElevenlabsEngine « api_key » ou dans la variable d'environnement ELEVENLABS_API_KEY)
- `mpv` installé sur votre système (essentiel pour le streaming audio mpeg, Elevenlabs ne fournit que du mpeg).

🔹 **Installation de `mpv` :**
- **macOS** :
```bash
brew install mpv
```

- **Linux et Windows** : Visitez [mpv.io](https://mpv.io/) pour obtenir des instructions d'installation.

### CoquiEngine

Fournit une synthèse vocale neuronale locale de haute qualité avec clonage de la voix.

Télécharge d'abord un modèle de synthèse vocale neuronale. Dans la plupart des cas, il doit être suffisamment rapide pour la synthèse en temps réel à l'aide du GPU. Nécessite environ 4 à 5 Go de VRAM.

- pour cloner une voix, soumettez le nom de fichier d'un fichier wave contenant la voix source comme paramètre « voix » au constructeur CoquiEngine
- le clonage de voix fonctionne mieux avec un fichier WAV mono 16 bits 22050 Hz contenant un échantillon court (~5-30 sec)

Sur la plupart des systèmes, la prise en charge du GPU sera nécessaire pour fonctionner suffisamment rapidement en temps réel, sinon vous subirez des bégaiements.

## Démarrage rapide

Voici un exemple d'utilisation de base :

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine() # replace with your TTS engine
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
```

## Feed Text

Vous pouvez alimenter des chaînes individuelles :

```python
stream.feed("Hello, this is a sentence.")
```

Vous pouvez également alimenter des générateurs et des itérateurs de caractères pour le streaming en temps réel :

```python
def write(prompt: str):
for chunk in openai.ChatCompletion.create(
model="gpt-3.5-turbo",
messages=[{"role": "user", "content" : prompt}],
stream=True
):
if (text_chunk := chunk["choices"][0]["delta"].get("content")) is not None:
yield text_chunk

text_stream = write("Un discours relaxant en trois phrases.")

stream.feed(text_stream)
```

```python
char_iterator = iter("Diffusion en continu de ce caractère par caractère.")
stream.feed(char_iterator)
```

## Lecture

De manière asynchrone :

```python
stream.play_async()
while stream.is_playing():
time.sleep(0.1)
```

De manière synchrone :

```python
stream.play()
```

## Test de la bibliothèque

Le sous-répertoire test contient un ensemble de scripts pour vous aider à évaluer et à comprendre les capacités de la bibliothèque RealtimeTTS.

Notez que la plupart des tests reposent toujours sur l'"ancienne" API OpenAI (<1.0.0). L'utilisation de la nouvelle API OpenAI est démontrée dans openai_1.0_test.py.

- **simple_test.py**
- **Description** : Une démonstration de style « hello world » de l'utilisation la plus simple de la bibliothèque.

- **complex_test.py**
- **Description** : Une démonstration complète présentant la plupart des fonctionnalités fournies par la bibliothèque.

- **coqui_test.py**
- **Description** : Test du moteur TTS local de coqui.

- **translator.py**
- **Dépendances** : Exécutez `pip install openai realtimestt`.
- **Description** : Traductions en temps réel dans six langues différentes.

- **openai_voice_interface.py**
- **Dépendances** : Exécutez `pip install openai realtimestt`.
- **Description** : Interface utilisateur activée par mot de réveil et basée sur la voix pour l'API OpenAI.

- **advanced_talk.py**
- **Dépendances** : Exécutez `pip install openai keyboard realtimestt`.
- **Description** : Choisissez le moteur TTS et la voix avant de démarrer la conversation AI.

- **minimalistic_talkbot.py**
- **Dépendances** : Exécutez `pip install openai realtimestt`.
- **Description** : Un talkbot de base en 20 lignes de code.

- **simple_llm_test.py**
- **Dépendances** : Exécutez `pip install op
  enai`.
- **Description** : Démonstration simple de la manière d'intégrer la bibliothèque à de grands modèles de langage (LLM).

- **test_callbacks.py**
- **Dépendances** : Exécutez `pip install openai`.
- **Description** : Présente les rappels et vous permet de vérifier les temps de latence dans un environnement d'application réel.

## Pause, reprise et arrêt

Mettre en pause le flux audio :

```python
stream.pause()
```

Reprendre un flux en pause :

```python
stream.resume()
```

Arrêter immédiatement le flux :

```python
stream.stop()
```

## Exigences expliquées

- **Version Python** :
- **Obligatoire** : Python >= 3.9, < 3.13
- **Raison** : La bibliothèque dépend de la bibliothèque GitHub « TTS » de coqui, qui nécessite des versions Python dans cette plage.

- **PyAudio** : pour créer un flux audio de sortie

- **stream2sentence** : pour diviser le flux de texte entrant en phrases

- **pyttsx3** : moteur de conversion texte-parole du système

- **pydub** : pour convertir les formats de morceaux audio

- **azure-cognitiveservices-speech** : moteur de conversion texte-parole Azure

- **elevenlabs** : moteur de conversion texte-parole Elevenlabs

- **coqui-TTS** : bibliothèque de synthèse vocale XTTS de Coqui pour une synthèse vocale neuronale locale de haute qualité

Un grand merci à [Idiap Research Institute](https://github.com/idiap) pour avoir maintenu un [fork de coqui tts](https://github.com/idiap/coqui-ai-TTS).

- **openai** : pour interagir avec l'API TTS d'OpenAI

- **gtts** : conversion de texte en parole de Google Translate

## Configuration

### Paramètres d'initialisation pour `TextToAudioStream`

Lorsque vous initialisez la classe `TextToAudioStream`, vous disposez de diverses options pour personnaliser son comportement. Voici les paramètres disponibles :

#### `engine` (BaseEngine)
- **Type** : BaseEngine
- **Obligatoire** : Oui
- **Description** : Le moteur sous-jacent responsable de la synthèse texte-audio. Vous devez fournir une instance de `BaseEngine` ou de sa sous-classe pour activer la synthèse audio.

#### `on_text_stream_start` (callable)
- **Type** : Fonction appelable
- **Obligatoire** : Non
- **Description** : Cette fonction de rappel facultative est déclenchée lorsque le flux de texte démarre. Utilisez-la pour toute configuration ou journalisation dont vous pourriez avoir besoin.

#### `on_text_stream_stop` (callable)
- **Type** : Fonction appelable
- **Obligatoire** : Non
- **Description** : Cette fonction de rappel facultative est activée lorsque le flux de texte se termine. Vous pouvez l'utiliser pour les tâches de nettoyage ou de journalisation.

#### `on_audio_stream_start` (callable)
- **Type** : Fonction appelable
- **Obligatoire** : Non
- **Description** : Cette fonction de rappel facultative est invoquée lorsque le flux audio démarre. Utile pour les mises à jour de l'interface utilisateur ou la journalisation des événements.

#### `on_audio_stream_stop` (callable)
- **Type** : Fonction appelable
- **Obligatoire** : Non
- **Description** : Cette fonction de rappel facultative est appelée lorsque le flux audio s'arrête. Idéal pour le nettoyage des ressources ou les tâches de post-traitement.

#### `on_character` (callable)
- **Type** : Fonction appelable
- **Obligatoire** : Non
- **Description** : Cette fonction de rappel facultative est appelée lorsqu'un seul caractère est traité.

#### `output_device_index` (int)
- **Type** : Entier
- **Obligatoire** : Non
- **Par défaut** : Aucun
- **Description** : Spécifie l'index du périphérique de sortie à utiliser. None utilise le périphérique par défaut.

#### `tokenizer` (string)
- **Type** : String
- **Obligatoire** : Non
- **Par défaut** : nltk
- **Description** : Tokenizer à utiliser pour le fractionnement de phrases (actuellement, « nltk » et « stanza » sont pris en charge).

#### `language` (string)
- **Type**: String
- **Required**: No
- **Default**: fr
- **Description**: Langue à utiliser pour le fractionnement des phrases.

#### `muted` (bool)
- **Type**: Bool
- **Required**: No
- **Default**: False
- **Description**: Paramètre global muted. Si True, aucun flux pyAudio ne sera ouvert. Désactive la lecture audio via les haut-parleurs locaux (au cas où vous voudriez synthétiser dans un fichier ou traiter des morceaux audio) et remplace le paramètre de mise en sourdine des paramètres de lecture.

#### `level` (int)
- **Type**: Integer
- **Required**: No
- **Default**: `logging.WARNING`
- **Description**: Définit le niveau de journalisation pour le logger interne. Il peut s'agir de n'importe quelle constante entière du module `logging` intégré de Python.

#### Exemple d'utilisation :

```python
engine = YourEngine() # Remplacez par votre moteur
stream = TextToAudioStream(
engine=engine,
on_text_stream_start=my_text_start_func,
on_text_stream_stop=my_text_stop_func,
on_audio_stream_start=my_audio_start_func,
on_audio_stream_stop=my_audio_stop_func,
level=logging.INFO
)
```

### Méthodes

#### `play` et `play_async`

Ces méthodes sont responsables de l'exécution de la synthèse texte-audio et de la lecture du flux audio. La différence est que `play` est une fonction de blocage, tandis que `play_async` s'exécute dans un thread séparé, permettant à d'autres opérations de se poursuivre.

##### Paramètres :

###### `fast_sentence_fragment` (bool)
- **Par défaut** : `True`
- **Description** : Lorsqu'elle est définie sur `True`, la méthode privilégie la vitesse, en générant et en lisant les fragments de phrases plus rapidement. Ceci est utile pour les applications où la latence est importante.

###### `fast_sentence_fragment_allsentences` (bool)
- **Par défaut** : `False`
- **Description** : lorsqu'il est défini sur « True », applique le traitement rapide des fragments de phrases à toutes les phrases, pas seulement à la première.

###### `fast_sentence_fragment_allsentences_multiple` (bool)
- **Par défaut** : « False`
- **Description** : lorsqu'il est défini sur « True », permet de générer plusieurs fragments de phrases au lieu d'un seul.

###### `buffer_threshold_seconds` (float)
- **Par défaut** : « 0.0 »
- **Description** : spécifie le temps en secondes pour le seuil de mise en mémoire tampon, qui a un impact sur la fluidité et la continuité de la lecture audio.

- **Fonctionnement** : avant de synthétiser une nouvelle phrase, le système vérifie s'il reste plus de matériel audio dans la mémoire tampon que le temps spécifié par `buffer_threshold_seconds`. Si tel est le cas, il récupère une autre phrase du générateur de texte, en supposant qu'il peut récupérer et synthétiser cette nouvelle phrase dans la fenêtre temporelle fournie par l'audio restant dans la mémoire tampon. Ce processus permet au moteur de synthèse vocale d'avoir plus de contexte pour une meilleure synthèse, améliorant ainsi l'expérience utilisateur.

Une valeur plus élevée garantit qu'il y a plus d'audio pré-tamponné, réduisant ainsi la probabilité de silence ou de lacunes pendant la lecture. Si vous rencontrez des pauses ou des interruptions, pensez à augmenter cette valeur.

###### `minimum_sentence_length` (int)
- **Par défaut** : `10`
- **Description** : Définit la longueur minimale de caractères pour considérer une chaîne comme une phrase à synthétiser. Cela affecte la manière dont les fragments de texte sont traités et lus.

###### `minimum_first_fragment_length` (int)
- **Par défaut** : `10`
- **Description** : Le nombre minimal de caractères requis pour le premier fragment de phrase avant de produire le résultat.

###### `log_synthesized_text` (bool)
- **Default**: `False`
- **Description**: Lorsqu'il est activé, enregistre les fragments de texte au fur et à mesure qu'ils sont synthétisés en audio. Utile pour l'audit et le débogage.

###### `reset_generated_text` (bool)
- **Default**: `True`
- **Description**: Si True, réinitialise le texte généré avant le traitement.

###### `output_wavfile` (str)
- **Default**: `None`
- **Description**: Si défini, enregistre l'audio dans le fichier WAV spécifié.

###### `on_sentence_synthesized` (callable)
- **Default**: `None`
- **Description**: Une fonction de rappel qui est appelée après la synthèse d'un seul fragment de phrase.

###### `before_sentence_synthesized` (callable)
- **Default**: `None`
- **Description**: Une fonction de rappel qui est appelée avant qu'un seul fragment de phrase ne soit synthétisé.

###### `on_audio_chunk` (callable)
- **Default**: `None`
- **Description**: Fonction de rappel qui est appelée lorsqu'un seul fragment audio est prêt.

###### `tokenizer` (str)
- **Default**: `"nltk"`
- **Description**: Générateur de tokens à utiliser pour le fractionnement des phrases. Prend actuellement en charge "nltk" et "stanza".

###### `tokenize_sentences` (callable)
- **Default**: `None`
- **Description**: Une fonction personnalisée qui tokenise les phrases à partir du texte d'entrée. Vous pouvez fournir votre propre tokenizer léger si vous n'êtes pas satisfait de nltk et stanza. Il doit prendre le texte comme une chaîne et renvoyer les phrases divisées comme une liste de chaînes.

###### `language` (str)
- **Default**: `"en"`
- **Description**: Langue à utiliser pour le fractionnement des phrases.

###### `context_size` (int)
- **Default**: `12`
- **Description**: Le nombre de caractères utilisés pour établir le contexte pour la détection des limites de phrases. Un contexte plus large améliore la précision de la détection des limites de phrases.

###### `context_size_look_overhead` (int)
- **Default**: `12`
- **Description**: Taille de contexte supplémentaire pour regarder en avant lors de la détection des limites de phrases.

###### `muted` (bool)
- **Default**: `False`
- **Description**: Si True, désactive la lecture audio via les haut-parleurs locaux. Utile lorsque vous souhaitez synthétiser dans un fichier ou traiter des morceaux audio sans les lire.

###### `sentence_fragment_delimiters` (str)
- **Default**: `".?!;:,\n…)]}。-"`
- **Description**: Une chaîne de caractères considérés comme des délimiteurs de phrases.

###### `force_first_fragment_after_words` (int)
- **Default**: `15`
- **Description**: Le nombre de mots après lequel le premier fragment de phrase est forcé à être rendu.

### Installation de CUDA

Ces étapes sont recommandées pour ceux qui ont besoin de **meilleures performances** et qui ont un GPU NVIDIA compatible.

> **Remarque** : *pour vérifier si votre GPU NVIDIA prend en charge CUDA, visitez la [liste officielle des GPU CUDA](https://developer.nvidia.com/cuda-gpus).*

Pour utiliser une torche avec prise en charge via CUDA, veuillez suivre ces étapes :

> **Remarque** : *les installations plus récentes de Pytorch [peuvent](https://stackoverflow.com/a/77069523) (non vérifiées) ne plus nécessiter l'installation de Toolkit (et éventuellement de cuDNN).*

1. **Installez NVIDIA CUDA Toolkit** :
Par exemple, pour installer Toolkit 12.X, veuillez
- Visitez [Téléchargements NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads).
- Sélectionnez votre système d'exploitation, l'architecture de votre système et la version de votre système d'exploitation.
- Téléchargez et installez le logiciel.

ou pour installer Toolkit 11.8, veuillez
- Visitez [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive).
- Sélectionnez votre
- système d'exploitation, architecture système et version du système d'exploitation.
- Téléchargez et installez le logiciel.

2. **Installez NVIDIA cuDNN** :

Par exemple, pour installer cuDNN 8.7.0 pour CUDA 11.x, veuillez
- Visitez [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive).
- Cliquez sur « Télécharger cuDNN v8.7.0 (28 novembre 2022), pour CUDA 11.x ».
- Téléchargez et installez le logiciel.

3. **Installez ffmpeg** :

Vous pouvez télécharger un programme d'installation pour votre système d'exploitation à partir du [site Web ffmpeg](https://ffmpeg.org/download.html).

Ou utilisez un gestionnaire de paquets :

- **Sur Ubuntu ou Debian** :
```bash
sudo apt update && sudo apt install ffmpeg
```

- **Sur Arch Linux** :
```bash
sudo pacman -S ffmpeg
```

- **Sur MacOS avec Homebrew** ([https://brew.sh/](https://brew.sh/)) :
```bash
brew install ffmpeg
```

- **Sur Windows avec Chocolatey** ([https://chocolatey.org/](https://chocolatey.org/)) :
```bash
choco install ffmpeg
```

- **Sur Windows avec Scoop** ([https://scoop.sh/](https://scoop.sh/)) :
```bash
scoop install ffmpeg
```

4. **Installer PyTorch avec prise en charge CUDA** :

Pour mettre à niveau votre installation PyTorch afin d'activer la prise en charge GPU avec CUDA, suivez ces instructions en fonction de votre version spécifique de CUDA. Cela est utile si vous souhaitez améliorer les performances de RealtimeSTT avec les fonctionnalités CUDA.

- **Pour CUDA 11.8 :**

Pour mettre à jour PyTorch et Torchaudio afin de prendre en charge CUDA 11.8, utilisez les commandes suivantes :

```bash
pip install torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

- **Pour CUDA 12.X :**

Pour mettre à jour PyTorch et Torchaudio afin de prendre en charge CUDA 12.X, exécutez la commande suivante :

```bash
pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

Remplacez « 2.3.1 » par la version de PyTorch qui correspond à votre système et à vos exigences.

5. **Correction pour résoudre les problèmes de compatibilité** :
Si vous rencontrez des problèmes de compatibilité de bibliothèque, essayez de définir ces bibliothèques sur des versions fixes :

```bash
pip install networkx==2.8.8
pip install typing_extensions==4.8.0
pip install fsspec==2023.6.0
pip install imageio==2.31.6
pip install networkx==2.8.8
pip install numpy==1.24.3
pip install requests==2.31.0
```

## 💖 Remerciements

Un grand merci à l'équipe derrière [Coqui AI](https://coqui.ai/) - en particulier la brillante [Eren Gölge](https://github.com/erogol) - pour avoir été la première à nous fournir une synthèse locale de haute qualité avec une vitesse en temps réel et même une voix clonable !

Merci à [Pierre Nicolas Durette](https://github.com/pndurette) de nous avoir offert un tts gratuit à utiliser sans GPU en utilisant Google Translate avec sa bibliothèque python gtts.

## Contribution

Les contributions sont toujours les bienvenues (par exemple, PR pour ajouter un nouveau moteur).

## Informations sur la licence

### ❗ Remarque importante :
Bien que la source de cette bibliothèque soit open source, l'utilisation de nombreux moteurs dont elle dépend ne l'est pas : les fournisseurs de moteurs externes limitent souvent l'utilisation commerciale dans leurs plans gratuits. Cela signifie que les moteurs peuvent être utilisés pour des projets non commerciaux, mais l'utilisation commerciale nécessite un plan payant.

### Résumé des licences de moteur :

#### CoquiEngine
- **Licence** : Open source uniquement pour les projets non commerciaux.
- **Utilisation commerciale** : Nécessite un plan payant.
- **Détails** : [Licence CoquiEngine](https://coqui.ai/cpml)

#### ElevenlabsEngine
- **Licence** : Open source uniquement pour les projets non commerciaux.
- **Utilisation commerciale** : Disponible avec tous les forfaits payants.
- **Détails** : [Licence ElevenlabsEngine](https://help.elevenlabs.io/hc/en-us/articles/13313564601361-Puis-je-publier-le-contenu-que-je-génère-sur-la-plateforme-)

#### AzureEngine
- **Licence** : Open source uniquement pour les projets non commerciaux.
- **Utilisation commerciale** : Disponible à partir du niveau standard.
- **Détails** : [Licence AzureEngine](https://learn.microsoft.com/en-us/answers/questions/1192398/can-i-use-azure-text-to-speech-for-commercial-usag)

#### SystemEngine
- **Licence** : Mozilla Public License 2.0 et GNU Lesser General Public License (LGPL) version 3.0.
- **Utilisation commerciale** : autorisée sous cette licence.
- **Détails** : [Licence SystemEngine](https://github.com/nateshmbhat/pyttsx3/blob/master/LICENSE)

#### GTTSEngine
- **Licence** : licence MIT
- **Utilisation commerciale** : elle est sous licence MIT, elle devrait donc théoriquement être possible. Une certaine prudence peut être nécessaire car elle utilise la fonctionnalité vocale non documentée de Google Translate.
- **Détails** : [Licence MIT GTTS](https://github.com/pndurette/gTTS/blob/main/LICENSE)

#### OpenAIEngine
- **Licence** : veuillez lire les [Conditions d'utilisation d'OpenAI](https://openai.com/policies/terms-of-use)

**Avertissement** : il s'agit d'un résumé des licences telles qu'elles sont comprises au moment de la rédaction. Il ne s'agit pas d'un avis juridique. Veuillez lire et respecter les licences des différents fournisseurs de moteurs si vous
Vous prévoyez de les utiliser dans un projet.

## Contributeurs

<a href="https://github.com/traceloop/openllmetry/graphs/contributors">
<img alt="contributeurs" src="https://contrib.rocks/image?repo=KoljaB/RealtimeTTS"/>
</a>

## Auteur

Kolja Beigel
E-mail : kolja.beigel@web.de

<p align="center">
<a href="https://github.com/KoljaB/RealtimeTTS" target="_blank">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
</a>
&nbsp;&nbsp;&nbsp;
<a href="#realtimetts" target="_blank">
<img src="https://img.shields.io/badge/Retour%20en%20haut-000000?style=pour-le-badge" alt="Retour en haut">
</a>
</p>
