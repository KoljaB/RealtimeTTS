# Utilisation

## Démarrage rapide

Voici un exemple d'utilisation de base :

````(`python
depuis RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

moteur = SystemEngine () # remplacer par votre moteur TTS
flux = TextToAudioStream(moteur)
stream.feed("Bonjour le monde! Comment ça va aujourd'hui ?")
stream.play_async()
``

## Flux Texte

Vous pouvez alimenter des chaînes individuelles :

````(`python
stream.feed(« Bonjour, c'est une phrase. »)
``

Ou vous pouvez alimenter des générateurs et des itérateurs de caractères pour le streaming en temps réel :

````(`python
def write (prompt : str) :
    pour chunk en openai.ChatCompletion.create(
        modèle="gpt-3.5-turbo",
        messages=[{"role": "utilisateur", "contenu" : prompt}],
        stream=True
    ):
        si (text_chunk := chunk[« choix »][0][« delta »].get(« contenu »)) n'est pas Aucun :
            produire du texte_chunk

text_stream = write (« Un discours relaxant en trois phrases »)

stream.feed(text_stream)
``

````(`python
char_iterator = iter (« Diffusion de ce personnage par personnage »)
stream.feed (char_iterator)
``

##Layback

Asynchrone:

````(`python
stream.play_async()
pendant que stream.is_playing():
    temps.sommeil(0,1)
``

Synchronisé:

````(`python
stream.play()
``

## Tester la bibliothèque

Le sous-répertoire de test contient un ensemble de scripts pour vous aider à évaluer et comprendre les capacités de la bibliothèque RealtimeTTS.

Notez que la plupart des tests reposent toujours sur l'« ancienne » API OpenAI (<1.0.0). L'utilisation de la nouvelle API OpenAI est démontrée dans openai_1.0_test.py.

- **simple_test.py**
    - **Description** : Une démonstration de style « hello world » de l'usage le plus simple de la bibliothèque.

- **complex_test.py**
    - **Description** : Une démonstration complète présentant la plupart des fonctionnalités fournies par la bibliothèque.

- **coqui_test.py**
    - **Description** : Test du moteur local coqui TTS.

- **traducteur.py**
    - **Dépendances**: Exécuter `pip install openai realtimestt`.
    - **Description** : Traductions en temps réel dans six langues différentes.

- **openai_voice_interface.py**
    - **Dépendances**: Exécuter `pip install openai realtimestt`.
    - **Description** : Interface utilisateur activée par mot de réveil et basée sur la voix vers l'API OpenAI.

- **advanced_talk.py**
    - **Dépendances**: Exécuter `pip install openai keyboard realtimestt`.
    - **Description** : Choisissez le moteur et la voix TTS avant de démarrer la conversation sur l'IA.

- **_talkbot.py** minimaliste
    - **Dépendances**: Exécuter `pip install openai realtimestt`.
    - **Description** : Un talkbot basique en 20 lignes de code.

- **simple_llm_test.py**
    - **Dépendances**: Exécuter `pip install openai`.
    - **Description** : Démonstration simple de la façon d'intégrer la bibliothèque avec de grands modèles de langage (LLM).

- **test_callbacks.py**
    - **Dépendances**: Exécuter `pip install openai`.
    - **Description** : présente les rappels et vous permet de vérifier les temps de latence dans un environnement d'application réel.

## Mettre en pause, reprendre et arrêter

Mettre en pause le flux audio :

````(`python
stream.pause()
``

Reprendre un flux en pause :

````(`python
stream.reprendre()
``

Arrêtez immédiatement le flux :

````(`python
stream.stop()
``

## Exigences expliquées

- **Version Python**:
  - **Obligatoire**: Python >= 3.9, < 3.13
  - **Raison** : La bibliothèque dépend de la bibliothèque GitHub « TTS » de coqui, qui nécessite des versions Python dans cette gamme.

- **PyAudio** : pour créer un flux audio de sortie

- **stream2sent** : pour diviser le flux de texte entrant en phrases

- **pyttsx3** : Moteur de conversion texte-parole du système

- **pydub** : pour convertir les formats de morceaux audio

- **azure-cognitiveservices-speech** : Moteur de conversion texte-parole azur

- **elevenlabs** : Moteur de conversion texte-parole Elevenlabs

- **coqui-TTS** : Bibliothèque de synthèse vocale XTTS de Coqui pour un TTS neuronal local de haute qualité

  Criez à [Idiap Research Institute](https://github.com/idiap) pour entretenir une [fourche de coqui tts](https://github.com/idiap/coqui-ai-TTS).

- **openai** : pour interagir avec l'API TTS d'OpenAI

- **gtts** : Google traduit la conversion texte-parole