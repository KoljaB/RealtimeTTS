# Uso

## Inicio Rápido

Aquí hay un ejemplo básico de uso:

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine() # replace with your TTS engine
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
```

## Alimentar Texto

Puede alimentar cadenas individuales:

```python
stream.feed("Hello, this is a sentence.")
```

O puede alimentar generadores e iteradores de caracteres para la transmisión en tiempo real:

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

## Reproducción

De forma asíncrona:

```python
stream.play_async()
while stream.is_playing():
    time.sleep(0.1)
```

De forma síncrona:

```python
stream.play()
```

## Prueba de la Biblioteca

El subdirectorio de pruebas contiene un conjunto de scripts para ayudarte a evaluar y comprender las capacidades de la biblioteca RealtimeTTS.

Ten en cuenta que la mayoría de las pruebas aún dependen de la API "antigua" de OpenAI (<1.0.0). El uso de la nueva API de OpenAI se demuestra en openai_1.0_test.py.

- **simple_test.py**
    - **Descripción**: Una demostración tipo "hola mundo" del uso más simple de la biblioteca.

- **complex_test.py**
    - **Descripción**: Una demostración completa que muestra la mayoría de las características proporcionadas por la biblioteca.

- **coqui_test.py**
    - **Descripción**: Prueba del motor local coqui TTS.

- **translator.py**
    - **Dependencias**: Ejecutar `pip install openai realtimestt`.
    - **Descripción**: Traducciones en tiempo real a seis idiomas diferentes.

- **openai_voice_interface.py**
    - **Dependencias**: Ejecutar `pip install openai realtimestt`.
    - **Descripción**: Interfaz de usuario activada por palabra clave y basada en voz para la API de OpenAI.

- **advanced_talk.py**
    - **Dependencias**: Ejecutar `pip install openai keyboard realtimestt`.
    - **Descripción**: Elija el motor TTS y la voz antes de iniciar la conversación con IA.

- **minimalistic_talkbot.py**
    - **Dependencias**: Ejecutar `pip install openai realtimestt`.
    - **Descripción**: Un talkbot básico en 20 líneas de código.

- **simple_llm_test.py**
    - **Dependencias**: Ejecutar `pip install openai`.
    - **Descripción**: Demostración simple de cómo integrar la biblioteca con modelos de lenguaje grande (LLMs).

- **test_callbacks.py**
    - **Dependencias**: Ejecutar `pip install openai`.
    - **Descripción**: Muestra los callbacks y te permite verificar los tiempos de latencia en un entorno de aplicación del mundo real.

## Pausar, Reanudar y Detener

Pausar el flujo de audio:

```python
stream.pause()
```

Reanudar un flujo pausado:

```python
stream.resume()
```

Detener el flujo inmediatamente:

```python
stream.stop()
```

## Requisitos Explicados

- **Versión de Python**:
  - **Requerido**: Python >= 3.9, < 3.13
  - **Razón**: La biblioteca depende de la biblioteca GitHub "TTS" de coqui, que requiere versiones de Python en este rango.

- **PyAudio**: para crear un flujo de audio de salida

- **stream2sentence**: para dividir el flujo de texto entrante en oraciones

- **pyttsx3**: Motor de conversión de texto a voz del sistema

- **pydub**: para convertir formatos de fragmentos de audio

- **azure-cognitiveservices-speech**: Motor de conversión de texto a voz de Azure

- **elevenlabs**: Motor de conversión de texto a voz de Elevenlabs

- **coqui-TTS**: Biblioteca de texto a voz XTTS de Coqui para TTS neuronal local de alta calidad

  Agradecimiento especial al [Instituto de Investigación Idiap](https://github.com/idiap) por mantener un [fork de coqui tts](https://github.com/idiap/coqui-ai-TTS).

- **openai**: para interactuar con la API TTS de OpenAI

- **gtts**: Conversión de texto a voz de Google translate