# Utilizzo

## Avvio Rapido

Ecco un esempio di utilizzo base:

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine() # sostituisci con il tuo motore TTS
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
```

## Inserimento del Testo

Puoi inserire singole stringhe:

```python
stream.feed("Hello, this is a sentence.")
```

Oppure puoi inserire generatori e iteratori di caratteri per lo streaming in tempo reale:

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

## Riproduzione

In modo asincrono:

```python
stream.play_async()
while stream.is_playing():
    time.sleep(0.1)
```

In modo sincrono:

```python
stream.play()
```

## Test della Libreria

La sottodirectory test contiene una serie di script per aiutarti a valutare e comprendere le capacità della libreria RealtimeTTS.

Nota che la maggior parte dei test si basa ancora sulla "vecchia" API OpenAI (<1.0.0). L'utilizzo della nuova API OpenAI è dimostrato in openai_1.0_test.py.

- **simple_test.py**
    - **Descrizione**: Una dimostrazione stile "hello world" dell'utilizzo più semplice della libreria.

- **complex_test.py**
    - **Descrizione**: Una dimostrazione completa che mostra la maggior parte delle funzionalità fornite dalla libreria.

- **coqui_test.py**
    - **Descrizione**: Test del motore TTS coqui locale.

- **translator.py**
    - **Dipendenze**: Esegui `pip install openai realtimestt`.
    - **Descrizione**: Traduzioni in tempo reale in sei lingue diverse.

- **openai_voice_interface.py**
    - **Dipendenze**: Esegui `pip install openai realtimestt`.
    - **Descrizione**: Interfaccia utente vocale attivata da parola chiave per l'API OpenAI.

- **advanced_talk.py**
    - **Dipendenze**: Esegui `pip install openai keyboard realtimestt`.
    - **Descrizione**: Scegli il motore TTS e la voce prima di iniziare la conversazione con l'IA.

- **minimalistic_talkbot.py**
    - **Dipendenze**: Esegui `pip install openai realtimestt`.
    - **Descrizione**: Un talkbot base in 20 righe di codice.

- **simple_llm_test.py**
    - **Dipendenze**: Esegui `pip install openai`.
    - **Descrizione**: Semplice dimostrazione di come integrare la libreria con i modelli linguistici di grandi dimensioni (LLM).

- **test_callbacks.py**
    - **Dipendenze**: Esegui `pip install openai`.
    - **Descrizione**: Mostra i callback e ti permette di verificare i tempi di latenza in un ambiente applicativo reale.

## Pausa, Ripresa e Stop

Metti in pausa lo stream audio:

```python
stream.pause()
```

Riprendi uno stream in pausa:

```python
stream.resume()
```

Ferma immediatamente lo stream:

```python
stream.stop()
```

## Requisiti Spiegati

- **Versione Python**:
  - **Richiesto**: Python >= 3.9, < 3.13
  - **Motivo**: La libreria dipende dalla libreria GitHub "TTS" di coqui, che richiede versioni Python in questo intervallo.

- **PyAudio**: per creare uno stream audio di output

- **stream2sentence**: per dividere il flusso di testo in ingresso in frasi

- **pyttsx3**: Motore di conversione text-to-speech di sistema

- **pydub**: per convertire i formati dei chunk audio

- **azure-cognitiveservices-speech**: Motore di conversione text-to-speech di Azure

- **elevenlabs**: Motore di conversione text-to-speech di Elevenlabs

- **coqui-TTS**: Libreria text-to-speech XTTS di Coqui per TTS neurale locale di alta qualità

  Un ringraziamento speciale a [Idiap Research Institute](https://github.com/idiap) per il mantenimento di un [fork di coqui tts](https://github.com/idiap/coqui-ai-TTS).

- **openai**: per interagire con l'API TTS di OpenAI

- **gtts**: Conversione text-to-speech di Google translate