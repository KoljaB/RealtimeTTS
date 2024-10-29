# الاستخدام

## البداية السريعة

إليك مثال بسيط للاستخدام:

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine() # replace with your TTS engine
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
```

نص التغذية

يمكنك إدخال سلاسل فردية:

```python
stream.feed("مرحبًا، هذه جملة.")
```

أو يمكنك تزويد المولدات ومكررات الشخصيات للبث المباشر:

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

## التشغيل

بشكل غير متزامن:

```python
stream.play_async()
while stream.is_playing():
    time.sleep(0.1)
```

بالتزامن:

```python
stream.play()
```

## اختبار المكتبة

تحتوي الدليل الفرعي للاختبار على مجموعة من السكربتات لمساعدتك في تقييم وفهم قدرات مكتبة RealtimeTTS.

يرجى ملاحظة أن معظم الاختبارات لا تزال تعتمد على واجهة برمجة التطبيقات "القديمة" من OpenAI (<1.0.0). يتم توضيح استخدام واجهة برمجة التطبيقات الجديدة من OpenAI في openai_1.0_test.py.

- **simple_test.py**
    - **الوصف**: عرض توضيحي بأسلوب "مرحبا بالعالم" لأبسط استخدامات المكتبة.

- **complex_test.py**
    - **الوصف**: عرض شامل يوضح معظم الميزات التي توفرها المكتبة.

- **coqui_test.py**
    - **الوصف**: اختبار لمحرك تحويل النص إلى كلام المحلي coqui.

- **translator.py**
    - **التبعيات**: شغل `pip install openai realtimestt`.
    - **الوصف**: ترجمات فورية إلى ست لغات مختلفة.

- **openai_voice_interface.py**
    - **التبعيات**: شغل `pip install openai realtimestt`.
    السياق: - **الوصف**:  
النص للترجمة: - **الوصف**: تم تفعيل كلمة الاستيقاظ وواجهة المستخدم المعتمدة على الصوت لواجهة برمجة تطبيقات OpenAI.

- **advanced_talk.py**
    - **التبعيات**: شغل الأمر `pip install openai keyboard realtimestt`.
    - **الوصف**: اختر محرك تحويل النص إلى كلام والصوت قبل بدء المحادثة مع الذكاء الاصطناعي.

- **minimalistic_talkbot.py**
    - **المتطلبات**: قم بتشغيل `pip install openai realtimestt`.
    - **الوصف**: روبوت محادثة بسيط في 20 سطر من الشيفرة.

- **simple_llm_test.py**
    - **التبعيات**: شغّل `pip install openai`.
    - **الوصف**: عرض بسيط لكيفية دمج المكتبة مع نماذج اللغة الكبيرة (LLMs).

- **test_callbacks.py**
    - **التبعيات**: شغل `pip install openai`.
    السياق: - **الوصف**:  
النص للترجمة: - **الوصف**: يعرض الاستدعاءات ويتيح لك التحقق من أوقات الكمون في بيئة تطبيقات العالم الحقيقي.

## إيقاف مؤقت، استئناف وإيقاف

أوقف بث الصوت:

```python
stream.pause()
```

استئناف بث متوقف:

```python
stream.resume()
```

أوقف البث فورًا:

```python
stream.stop()
```

## متطلبات موضحة

- **إصدار بايثون**:
  - **المطلوب**: بايثون >= 3.9, < 3.13
  - **السبب**: المكتبة تعتمد على مكتبة GitHub "TTS" من coqui، والتي تتطلب إصدارات بايثون في هذا النطاق.

- **PyAudio**: لإنشاء دفق صوتي خارجي

- **stream2sentence**: لتقسيم تدفق النص الوارد إلى جمل

- **pyttsx3**: محرك تحويل النص إلى كلام للنظام

- **pydub**: لتحويل تنسيقات مقاطع الصوت

- **azure-cognitiveservices-speech**: محرك تحويل النص إلى كلام من Azure

- **elevenlabs**: محرك تحويل النص إلى كلام من Elevenlabs

- **coqui-TTS**: مكتبة Coqui XTTS لتحويل النص إلى كلام بجودة عالية باستخدام الشبكات العصبية المحلية

  تحية إلى [معهد أبحاث إيديا](https://github.com/idiap) على الحفاظ على [نسخة مفرعة من coqui tts](https://github.com/idiap/coqui-ai-TTS).

- **openai**: للتفاعل مع واجهة برمجة التطبيقات لتحويل النص إلى كلام من OpenAI

- **gtts**: تحويل النص إلى كلام باستخدام Google Translate