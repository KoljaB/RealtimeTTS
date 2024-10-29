# 사용 방법

## 빠른 시작

기본적인 사용 예는 다음과 같습니다:

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine


engine = SystemEngine() # TTS 엔진을 지정
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
```
## 텍스트 피드

개별 문자열을 입력할 수 있습니다:

```python
stream.feed("안녕하세요, 이것은 문장입니다.")
```

실시간 스트리밍용 제너레이터나 문자 이터레이터를 피드하는 것도 가능합니다:

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

## 재생

비동기 재생:

```python
stream.play_async()
while stream.is_playing():
    time.sleep(0.1)
```

동기 재생:

```python
stream.play()
```

## 라이브러리 테스트

test 서브디렉토리에는 RealtimeTTS 라이브러리의 기능을 평가하고 이해하기 위한 스크립트가 포함되어 있습니다.

오래된 OpenAI API (<1.0.0)에 의존하는 테스트가 많기 때문에, 새로운 OpenAI API의 사용 예는 openai_1.0_test.py에서 확인할 수 있습니다.

- **simple_test.py**
    - **설명**: 라이브러리의 가장 간단한 사용법을 보여주는 "Hello World" 스타일의 데모.

- **complex_test.py**
    - **설명**: 라이브러리의 대부분 기능을 포괄하는 종합적인 데모.

- **coqui_test.py**
    - **설명**: 로컬 Coqui TTS 엔진의 테스트.

- **translator.py**
    - **의존성**: `pip install openai realtimestt`를 실행하세요.
    - **설명**: 6개의 다른 언어로 실시간 번역.

- **openai_voice_interface.py**
    - **의존성**: `pip install openai realtimestt`를 실행하십시오.
    - **설명**: 시작 단어로 시작되는 음성 기반의 OpenAI API 인터페이스.

- **advanced_talk.py**
    - **의존성**: `pip install openai keyboard realtimestt`를 실행하십시오.
    - **설명**: TTS 엔진과 목소리를 선택하여 AI 대화를 시작하세요.

- **minimalistic_talkbot.py**
    - **의존성**: `pip install openai realtimestt` 실행.
    - **説明**: 20行のコードで作成されたシンプルなトークボット。

- **simple_llm_test.py**
    - **依存関係**: `pip install openai`。
    - **説明**: ラージランゲージモデル（LLM）との統合の簡単なデモ。

- **test_callbacks.py**
    - **依存関係**: `pip install openai`。
    - **説明**: コールバックを紹介し、実環境での待機時間をチェック可能。

## 一時停止、再開 & 停止

オーディオストリームを一時停止：

```python
stream.pause()
```

一時停止したストリームを再開：

```python
stream.resume()
```

ストリームをすぐに停止：

```python
stream.stop()
```

## 必要要件の説明

- **Pythonバージョン**:
  - **必須**: Python >= 3.9, < 3.13
  - **理由**: ライブラリはCoquiのGitHubライブラリ「TTS」に依存しており、Pythonのこのバージョン範囲が必要です。

- **PyAudio**: 出力オーディオストリームを作成するため

- **stream2sentence**: 入力されるテキストストリームを文単位に分割するため

- **pyttsx3**: システムテキスト読み上げエンジン

- **pydub**: オーディオチャンクの形式変換用

- **azure-cognitiveservices-speech**: Azureテキスト読み上げエンジン

- **elevenlabs**: Elevenlabsテキスト読み上げエンジン

- **coqui-TTS**: 高品質なローカルニューラルTTS用CoquiのXTTSテキスト読み上げライブラリ

  [Idiap研究所](https://github.com/idiap)が管理する[Coqui TTSのフォーク](https://github.com/idiap/coqui-ai-TTS)に感謝。

- **openai**: OpenAIのTTS APIとのインタラクション用

- **gtts**: Google翻訳テキスト読み上げ変換

