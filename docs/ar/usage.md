# طريقة الاستخدام

## البداية السريعة

الأمثلة الأساسية للاستخدام هي كما يلي:

من RealtimeTTS استيراد TextToAudioStream، SystemEngine، AzureEngine، ElevenlabsEngine

المحرك = SystemEngine() # تحديد محرك TTS
التدفق = TextToAudioStream(engine)
stream.feed("مرحباً بالعالم! كيف حالك اليوم؟
stream.play_async()

## نص التغذية

يمكنك إدخال سلاسل فردية:

```python
stream.feed("مرحبًا، هذه جملة.")
```

من الممكن أيضًا تزويد مولدات أو مكررات النصوص للبث المباشر:

```python
def write(prompt: str):
    for chunk in openai.
```إكمال الدردشة.إنشاء(
        النموذج="gpt-3.5-turbo",
        الرسائل=[{"الدور": "المستخدم", "المحتوى" : prompt}],
        التدفق=True
    ):
        إذا (text_chunk := chunk["choices"]النص للترجمة: [0]["دلتا"].get("content")) ليس None:
            إنتاج جزء_النص

text_stream = كتابةخطاب غير رسمي من ثلاث جمل.

stream.feed(text_stream)

```python
char_iterator = iter("سأقوم ببث هذا حرفًا بحرف.")
```
stream.feed(char_iterator)

## إعادة التشغيل

التشغيل غير المتزامن:

```python
stream.play_async()
while stream.is_playing():
    time.sleep(0.1)
```

التشغيل المتزامن:

```python
stream.play()
```

## اختبار المكتبة

تحتوي الدليل الفرعي test على نصوص لتقييم وفهم وظائف مكتبة RealtimeTTS.

نظرًا لأن العديد من الاختبارات تعتمد على واجهة برمجة التطبيقات القديمة لـ OpenAI (<1.0.0)، يمكن الاطلاع على أمثلة استخدام واجهة برمجة التطبيقات الجديدة في openai_1.0_test.py.

- **simple_test.py**
    - **الوصف**: عرض توضيحي بأسلوب "Hello World" يوضح أبسط استخدام للمكتبة.

- **complex_test.py**
    - **الوصف**: عرض شامل يغطي معظم ميزات المكتبة.

- **coqui_test.py**
    - **الوصف**: اختبار محرك Coqui TTS المحلي.

- **translator.py**
    - **اعتماديات**: نفذ الأمر `pip install openai realtimestt`.
    - **الوصف**: ترجمة فورية إلى ست لغات مختلفة.

- **openai_voice_interface.py**
    - **اعتماديات**: نفذ الأمر `pip install openai realtimestt`.
    - **الوصف**: واجهة برمجة تطبيقات OpenAI الصوتية التي تبدأ بكلمة تنشيط.

- **advanced_talk.py**
    - **الاعتماديات**: نفذ `pip install openai keyboard realtimestt`.
    - **الوصف**: اختر محرك TTS والصوت وابدأ المحادثة مع الذكاء الاصطناعي.

- **minimalistic_talkbot.py**
    - **اعتماديات**: نفذ `pip install openai realtimestt`.
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

