
# 使用方法

## クイックスタート

基本的な使用例は以下の通りです：

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine() # TTSエンジンを指定
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
```

## テキストのフィード

個々の文字列をフィードすることができます：

```python
stream.feed("Hello, this is a sentence.")
```

リアルタイムストリーミング用のジェネレータや文字イテレータをフィードすることも可能です：

```python
def write(prompt: str):
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content" : prompt}],
        stream=True
    ):
        if (text_chunk := chunk["choices"][0]["delta"].get("content")) is not None:
            yield text_chunk

text_stream = write("3文のリラックスしたスピーチ。")

stream.feed(text_stream)
```

```python
char_iterator = iter("文字ごとにこれをストリーミングします。")
stream.feed(char_iterator)
```

## 再生

非同期再生：

```python
stream.play_async()
while stream.is_playing():
    time.sleep(0.1)
```

同期再生：

```python
stream.play()
```

## ライブラリのテスト

testサブディレクトリには、RealtimeTTSライブラリの機能を評価および理解するためのスクリプトが含まれています。

古いOpenAI API (<1.0.0) に依存しているテストが多いため、新しいOpenAI APIの使用例はopenai_1.0_test.pyで確認できます。

- **simple_test.py**
    - **説明**: ライブラリの最も簡単な使用法を示す「Hello World」スタイルのデモ。

- **complex_test.py**
    - **説明**: ライブラリの大半の機能を網羅した包括的なデモ。

- **coqui_test.py**
    - **説明**: ローカルのCoqui TTSエンジンのテスト。

- **translator.py**
    - **依存関係**: `pip install openai realtimestt` を実行。
    - **説明**: 6つの異なる言語へのリアルタイム翻訳。

- **openai_voice_interface.py**
    - **依存関係**: `pip install openai realtimestt` を実行。
    - **説明**: 起動ワードで開始される音声ベースのOpenAI APIインターフェイス。

- **advanced_talk.py**
    - **依存関係**: `pip install openai keyboard realtimestt` を実行。
    - **説明**: TTSエンジンと声を選んでAI会話を開始。

- **minimalistic_talkbot.py**
    - **依存関係**: `pip install openai realtimestt` を実行。
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
