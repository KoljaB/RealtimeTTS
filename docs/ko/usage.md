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
## 텍스트 입력

개별 문자열을 입력할 수 있습니다:

```python
stream.feed("Hello, this is a sentence.")
```

또는, 실시간 스트리밍을 위한 제너레이터나 문자 이터레이터를 입력하는 것도 가능합니다:

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

비동기적 재생:

```python
stream.play_async()
while stream.is_playing():
    time.sleep(0.1)
```

동기적 재생:

```python
stream.play()
```

## 라이브러리 테스트

test 하위 디렉토리에는 RealtimeTTS 라이브러리의 기능을 평가하고 이해하기 위한 스크립트가 포함되어 있습니다.

오래된 OpenAI API (<1.0.0)에 의존하는 테스트가 많기 때문에, 새로운 OpenAI API의 사용 예는 openai_1.0_test.py에서 확인할 수 있습니다.

- **simple_test.py**
    - **설명**: 라이브러리의 가장 간단한 사용법을 보여주는 "Hello World" 스타일의 데모.

- **complex_test.py**
    - **설명**: 라이브러리의 대부분 기능을 포괄하는 종합적인 데모.

- **coqui_test.py**
    - **설명**: 로컬 Coqui TTS 엔진의 테스트.

- **translator.py**
    - **의존성**: `pip install openai realtimestt`를 실행
    - **설명**: 6개의 다른 언어로 실시간 번역.

- **openai_voice_interface.py**
    - **의존성**: `pip install openai realtimestt`를 실행
    - **설명**: 시작 단어로 시작되는 음성 기반의 OpenAI API 인터페이스.

- **advanced_talk.py**
    - **의존성**: `pip install openai keyboard realtimestt`를 실행
    - **설명**: AI 대화를 시작하기 전에 TTS 엔진과 음성을 선택하세요.

- **minimalistic_talkbot.py**
    - **의존성**: `pip install openai realtimestt`를 실행
    - **설명**: 20줄로 만든 기본 대화 봇.

- **simple_llm_test.py**
    - **의존성**: `pip install openai`를 실행
    - **설명**: 대형 언어 모델(LLM)과 라이브러리를 통합하는 방법을 간단히 시연합니다.

- **test_callbacks.py**
    - **의존성**: `pip install openai`。
    - **설명**: 콜백을 보여주고 실제 애플리케이션 환경에서 대기 시간을 확인할 수 있습니다.

## 일시정지, 재개 및 중지

오디오 스트림 일시정지：

```python
stream.pause()
```

오디오 스트림 재게：

```python
stream.resume()
```

스트림 즉시 중지：

```python
stream.stop()
```

## 필수 요구사항 설명

- **파이썬 버전**:
  - **필수 사항**: Python >= 3.9, < 3.13
  - **이유**: 이 라이브러리는 Coqui의 GitHub 라이브러리 "TTS"에 의존하며, 해당 라이브러리는 제시된 범위의 Python 버전을 필요로 합니다.

- **PyAudio**: 출력 오디오 스트림을 생성을 위해 필요

- **stream2sentence**: 텍스트 스트림을 문장 단위로 분할하기 위해 필요

- **pyttsx3**: 시스템 텍스트-음성 변환 엔진

- **pydub**: 오디오를 청크 형식으로 변환하기 위해 필요

- **azure-cognitiveservices-speech**: Azure 텍스트-음성 변환 엔진

- **elevenlabs**: Elevenlabs 텍스트-음성 변환 엔진

- **coqui-TTS**: Coqui의 XTTS 고품질 로컬 신경 텍스트-음성 변환 라이브러리

   [Idiap Research Institute](https://github.com/idiap)에 감사의 인사를 전하며, [coqui tts의 포크](https://github.com/idiap/coqui-ai-TTS)를 유지 관리해 주신 점에 감사합니다.

- **openai**: OpenAI의 TTS API와 상호작용하기 위해 필요

- **gtts**: Google 번역 텍스트-음성 변환

