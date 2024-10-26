# 使用

## クイックスタート

基本的な使用例は次のとおりです。

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine
```

engine = SystemEngine() # あなたのTTSエンジンに置き換えてください
ストリーム = TextToAudioStream(engine)
stream.feed("こんにちは、世界！") 今日はどうですか？
stream.play_async()

## フィードテキスト

個別の文字列を入力できます：

```python
stream.feed("こんにちは、これは文です。")
```

または、リアルタイムストリーミングのためにジェネレーターやキャラクターイテレーターを使用することもできます。

```python
def write(prompt: str):
    for chunk in openai.
```チャット完了。作成(
        モデル="gpt-3.5-turbo",
        メッセージ=[{"役割": "ユーザー", "内容": プロンプト}],
        ストリーム=True
    ):
        もし (text_chunk := chunk["choices"]テキストを翻訳してください。["デルタ"].get("content")) は None ではない:
            テキストチャンクを生成

テキストストリーム = 書き込み("A three-sentence relaxing speech.")

ストリームにテキストストリームをフィードする

```python
char_iterator = iter("文字を1文字ずつストリーミングします。")
stream.feed(char_iterator)
```

## 再生

非同期で:

```python
stream.play_async()
while stream.is_playing():
    time.sleep(0.1)
```

同時に:

```python
stream.play()
```

## ライブラリのテスト

テストサブディレクトリには、RealtimeTTSライブラリの機能を評価し理解するための一連のスクリプトが含まれています。

ほとんどのテストはまだ「古い」OpenAI API（<1.0.0）に依存していることに注意してください。 新しいOpenAI APIの使用方法はopenai_1.0_test.pyで示されています。

- **simple_test.py**
    - **説明**: ライブラリの最も簡単な使い方を示す「ハローワールド」スタイルのデモ。

- **complex_test.py**
    - **説明**: ライブラリが提供するほとんどの機能を紹介する包括的なデモ。

- **coqui_test.py**
    - **説明**: ローカルのcoqui TTSエンジンのテスト。

- **translator.py**
    - **依存関係**: `pip install openai realtimestt`を実行してください。
    - **説明**: 6つの異なる言語へのリアルタイム翻訳。

- **openai_voice_interface.py**
    - **依存関係**: `pip install openai realtimestt` を実行してください。
    - **説明**: ウェイクワードがアクティブになり、OpenAI APIへの音声ベースのユーザーインターフェース。

- **advanced_talk.py**
    - **依存関係**: `pip install openai keyboard realtimestt`を実行してください。
    - **説明**: AI会話を始める前に、TTSエンジンと声を選択してください。

- **minimalistic_talkbot.py**
    - **依存関係**: `pip install openai realtimestt`を実行してください。
    - **説明**: 20行のコードで作る基本的なトークボット。

- **simple_llm_test.py**
    - **依存関係**: `pip install openai`を実行してください。
    - **説明**: ライブラリを大規模言語モデルに統合する簡単なデモ (LLMs).

- **test_callbacks.py**
    - **依存関係**: `pip install openai`を実行してください。
    - **説明**: コールバックを紹介し、実際のアプリケーション環境でレイテンシー時間を確認できるようにします。

## 一時停止、再開、停止

オーディオストリームを一時停止:

```python
stream.pause()
```

一時停止中のストリームを再開する:

```python
stream.resume()
```

すぐに配信を停止してください。

```python
stream.stop()
```

## 要件の説明

- **Pythonバージョン**:
  - **必要条件**: Python >= 3.9, < 3.13
  - **理由**: ライブラリは、coquiのGitHubライブラリ「TTS」に依存しており、これにはこの範囲のPythonバージョンが必要です。

- **PyAudio**: 出力オーディオストリームを作成するために

- **stream2sentence**: 入力されたテキストストリームを文に分割する

- **pyttsx3**: システム音声合成エンジン

- **pydub**: オーディオチャンクのフォーマットを変換するために

- **azure-cognitiveservices-speech**: Azure テキスト読み上げエンジン

- **elevenlabs**: Elevenlabs テキスト読み上げエンジン

- **coqui-TTS**: 高品質なローカルニューラルTTSのためのCoquiのXTTSテキスト読み上げライブラリ

  [Idiap Research Institute](https://github.com/idiap)が[coqui ttsのフォーク](https://github.com/idiap/coqui-ai-TTS)を維持していることに感謝します。

- **openai**: OpenAIのTTS APIと対話するために

- **gtts**: Google翻訳テキスト読み上げ変換

