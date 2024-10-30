# Uso

## Início Rápido

Aqui está um exemplo básico de uso:

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine() # replace with your TTS engine
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
```

## Texto de Alimentação

Você pode alimentar strings individuais:

```python
stream.feed("Hello, this is a sentence.")
```

Ou você pode alimentar geradores e iteradores de caracteres para streaming em tempo real:

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

## Reprodução

Assincronamente:

```python
stream.play_async()
while stream.is_playing():
    time.sleep(0.1)
```

Sincronamente:

```python
stream.play()
```

## Testando a Biblioteca

O subdiretório de testes contém um conjunto de scripts para ajudá-lo a avaliar e entender as capacidades da biblioteca RealtimeTTS.

Note que a maioria dos testes ainda depende da "antiga" API da OpenAI (<1.0.0). O uso da nova API OpenAI é demonstrado em openai_1.0_test.py.

- **simple_test.py**
    - **Descrição**: Uma demonstração no estilo "hello world" do uso mais simples da biblioteca.

- **complex_test.py**
    - **Descrição**: Uma demonstração abrangente que mostra a maioria dos recursos fornecidos pela biblioteca.

- **coqui_test.py**
    - **Descrição**: Teste do mecanismo TTS local do Coqui.

- **translator.py**
    - **Dependências**: Execute `pip install openai realtimestt`.
    - **Descrição**: Traduções em tempo real para seis idiomas diferentes.

- **openai_voice_interface.py**
    - **Dependências**: Execute `pip install openai realtimestt`.
    - **Descrição**: Interface de usuário baseada em voz ativada por palavra-chave para a API OpenAI.

- **advanced_talk.py**
    - **Dependências**: Execute `pip install openai keyboard realtimestt`.
    - **Descrição**: Escolha o mecanismo TTS e a voz antes de iniciar a conversa com a IA.

- **minimalistic_talkbot.py**
    - **Dependências**: Execute `pip install openai realtimestt`.
    - **Descrição**: Um talkbot básico em 20 linhas de código.

- **simple_llm_test.py**
    - **Dependências**: Execute `pip install openai`.
    - **Descrição**: Demonstração simples de como integrar a biblioteca com grandes modelos de linguagem (LLMs).

- **test_callbacks.py**
    - **Dependências**: Execute `pip install openai`.
    - **Descrição**: Demonstra os callbacks e permite que você verifique os tempos de latência em um ambiente de aplicação do mundo real.

## Pausar, Retomar e Parar

Pausar o fluxo de áudio:

```python
stream.pause()
```

Retome um stream pausado:

```python
stream.resume()
```

Pare o stream imediatamente:

```python
stream.stop()
```

## Requisitos Explicados

- **Versão do Python**:
  - **Requerido**: Python >= 3.9, < 3.13
  - **Motivo**: A biblioteca depende da biblioteca do GitHub "TTS" da coqui, que requer versões do Python dentro dessa faixa.

- **PyAudio**: para criar um fluxo de áudio de saída

- **stream2sentence**: para dividir o fluxo de texto recebido em sentenças

- **pyttsx3**: Motor de conversão de texto para fala do sistema

- **pydub**: para converter formatos de pedaços de áudio

- **azure-cognitiveservices-speech**: motor de conversão de texto para fala da Azure

- **elevenlabs**: Motor de conversão de texto para fala da Elevenlabs

- **coqui-TTS**: Biblioteca de conversão de texto em fala XTTS da Coqui para TTS neural local de alta qualidade

  Agradecimentos ao [Instituto de Pesquisa Idiap](https://github.com/idiap) por manter um [fork do coqui tts](https://github.com/idiap/coqui-ai-TTS).

- **openai**: para interagir com a API TTS da OpenAI

- **gtts**: Conversão de texto para fala do Google Translate