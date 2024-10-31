# Использование

## Быстрый старт

Вот базовый пример использования:

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine() # replace with your TTS engine
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
```

## Ввод текста

Вы можете подавать отдельные строки:

```python
stream.feed("Hello, this is a sentence.")
```

Или вы можете использовать генераторы и итераторы символов для потоковой передачи в реальном времени:

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

## Воспроизведение

Асинхронно:

```python
stream.play_async()
while stream.is_playing():
    time.sleep(0.1)
```

Синхронно:

```python
stream.play()
```

## Тестирование библиотеки

Подкаталог тестов содержит набор скриптов, которые помогут вам оценить и понять возможности библиотеки RealtimeTTS.

Обратите внимание, что большинство тестов все еще зависят от «старого» API OpenAI (<1.0.0). Использование нового API OpenAI демонстрируется в openai_1.0_test.py.

- **simple_test.py**
    - **Описание**: Демонстрация простейшего использования библиотеки в стиле "hello world".

- **complex_test.py**
    - **Описание**: Комплексная демонстрация, показывающая большинство функций, предоставляемых библиотекой.

- **coqui_test.py**
    - **Описание**: Тест локального движка TTS coqui.

- **translator.py**
    - **Зависимости**: Запустите `pip install openai realtimestt`.
    - **Описание**: Переводы в реальном времени на шесть разных языков.

- **openai_voice_interface.py**
    - **Зависимости**: Запустите `pip install openai realtimestt`.
    - **Описание**: Слово пробуждения активировано, и голосовой интерфейс пользователя для API OpenAI.

- **advanced_talk.py**
    - **Зависимости**: Запустите `pip install openai keyboard realtimestt`.
    - **Описание**: Выберите движок TTS и голос перед началом разговора с ИИ.

- **minimalistic_talkbot.py**
    - **Зависимости**: Запустите `pip install openai realtimestt`.
    - **Описание**: Простой разговорный бот на 20 строк кода.

- **simple_llm_test.py**
    - **Зависимости**: Запустите `pip install openai`.
    - **Описание**: Простая демонстрация того, как интегрировать библиотеку с крупными языковыми моделями (LLMs).

- **test_callbacks.py**
    - **Зависимости**: Запустите `pip install openai`.
    - **Описание**: Демонстрирует обратные вызовы и позволяет проверить время задержки в реальной среде приложения.

## Пауза, Возобновить и Остановить

Приостановите аудиопоток:

```python
stream.pause()
```

Возобновите приостановленный поток:

```python
stream.resume()
```

Немедленно остановите поток:

```python
stream.stop()
```

## Объяснение требований

- **Версия Python**:
  - **Требуется**: Python >= 3.9, < 3.13
  - **Причина**: Библиотека зависит от библиотеки "TTS" на GitHub от coqui, которая требует версии Python в этом диапазоне.

- **PyAudio**: для создания выходного аудиопотока

- **stream2sentence**: для разбивки входящего текстового потока на предложения

- **pyttsx3**: Системный движок преобразования текста в речь

- **pydub**: для преобразования форматов аудиофрагментов

- **azure-cognitiveservices-speech**: Движок преобразования текста в речь Azure

- **elevenlabs**: Движок преобразования текста в речь Elevenlabs

- **coqui-TTS**: Библиотека XTTS от Coqui для высококачественного локального нейронного TTS

  Приветствую [Исследовательский институт Idiap](https://github.com/idiap) за поддержку [форка coqui tts](https://github.com/idiap/coqui-ai-TTS).

- **openai**: для взаимодействия с API TTS от OpenAI

- **gtts**: Преобразование текста в речь от Google Translate