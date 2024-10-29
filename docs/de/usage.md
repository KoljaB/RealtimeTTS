# Verwendung

## Schnellstart

Hier ist ein grundlegendes Beispiel:

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine() # ersetzen Sie dies mit Ihrer TTS-Engine
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
```

## Text Einspeisung

Sie können einzelne Zeichenketten einspeisen:

```python
stream.feed("Hello, this is a sentence.")
```

Oder Sie können Generatoren und Zeichen-Iteratoren für Echtzeit-Streaming verwenden:

```python
def write(prompt: str):
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content" : prompt}],
        stream=True
    ):
        if (text_chunk := chunk["choices"][0]["delta"].get("content")) is not None:
            yield text_chunk

text_stream = write("A three-sentence relaxing speech.")

stream.feed(text_stream)
```

```python
char_iterator = iter("Streaming this character by character.")
stream.feed(char_iterator)
```

## Wiedergabe

Asynchron:

```python
stream.play_async()
while stream.is_playing():
    time.sleep(0.1)
```

Synchron:

```python
stream.play()
```

## Testen der Bibliothek

Das Test-Unterverzeichnis enthält eine Reihe von Skripten, die Ihnen helfen, die Fähigkeiten der RealtimeTTS-Bibliothek zu bewerten und zu verstehen.

Beachten Sie, dass die meisten Tests noch auf der "alten" OpenAI API (<1.0.0) basieren. Die Verwendung der neuen OpenAI API wird in openai_1.0_test.py demonstriert.

- **simple_test.py**
    - **Beschreibung**: Eine "Hello World"-ähnliche Demonstration der einfachsten Bibliotheksnutzung.

- **complex_test.py**
    - **Beschreibung**: Eine umfassende Demonstration der meisten Funktionen der Bibliothek.

- **coqui_test.py**
    - **Beschreibung**: Test der lokalen Coqui TTS-Engine.

- **translator.py**
    - **Abhängigkeiten**: Führen Sie `pip install openai realtimestt` aus.
    - **Beschreibung**: Echtzeitübersetzungen in sechs verschiedene Sprachen.

- **openai_voice_interface.py**
    - **Abhängigkeiten**: Führen Sie `pip install openai realtimestt` aus.
    - **Beschreibung**: Durch Aktivierungswort gesteuerte und sprachbasierte Benutzeroberfläche für die OpenAI API.

- **advanced_talk.py**
    - **Abhängigkeiten**: Führen Sie `pip install openai keyboard realtimestt` aus.
    - **Beschreibung**: Wählen Sie TTS-Engine und Stimme vor Beginn der KI-Konversation.

- **minimalistic_talkbot.py**
    - **Abhängigkeiten**: Führen Sie `pip install openai realtimestt` aus.
    - **Beschreibung**: Ein grundlegender Sprachbot in 20 Codezeilen.

- **simple_llm_test.py**
    - **Abhängigkeiten**: Führen Sie `pip install openai` aus.
    - **Beschreibung**: Einfache Demonstration der Integration der Bibliothek mit Large Language Models (LLMs).

- **test_callbacks.py**
    - **Abhängigkeiten**: Führen Sie `pip install openai` aus.
    - **Beschreibung**: Zeigt die Callbacks und lässt Sie die Latenzzeiten in einer realen Anwendungsumgebung überprüfen.

## Pause, Fortsetzen & Stoppen

Audiostream pausieren:

```python
stream.pause()
```

Pausierten Stream fortsetzen:

```python
stream.resume()
```

Stream sofort stoppen:

```python
stream.stop()
```

## Erläuterung der Anforderungen

- **Python Version**:
  - **Erforderlich**: Python >= 3.9, < 3.13
  - **Grund**: Die Bibliothek hängt von der GitHub-Bibliothek "TTS" von Coqui ab, die Python-Versionen in diesem Bereich erfordert.

- **PyAudio**: zur Erstellung eines Audio-Ausgabestreams

- **stream2sentence**: zum Aufteilen des eingehenden Textstreams in Sätze

- **pyttsx3**: System Text-to-Speech Konvertierungs-Engine

- **pydub**: zur Konvertierung von Audio-Chunk-Formaten

- **azure-cognitiveservices-speech**: Azure Text-to-Speech Konvertierungs-Engine

- **elevenlabs**: Elevenlabs Text-to-Speech Konvertierungs-Engine

- **coqui-TTS**: Coqui's XTTS Text-to-Speech Bibliothek für hochwertige lokale neuronale TTS

  Dank an das [Idiap Research Institute](https://github.com/idiap) für die Pflege eines [Forks von Coqui TTS](https://github.com/idiap/coqui-ai-TTS).

- **openai**: zur Interaktion mit der OpenAI TTS API

- **gtts**: Google Translate Text-to-Speech Konvertierung