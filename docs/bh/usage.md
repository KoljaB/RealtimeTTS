# उपयोग

## त्वरित शुरुआत

एहिजा एगो बेसिक उदाहरण बा:

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine() # अपना TTS-इंजन के साथ ए बदलें
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
```

## टेक्स्ट इनपुट

रउआँ व्यक्तिगत स्ट्रिंग्स के इनपुट कर सकतानी:

```python
stream.feed("Hello, this is a sentence.")
```

अथवा, रउआँ रियल टाइम-स्ट्रीमिंग खातिर जनरेटर आ कैरेक्टर-इटरेटर के उपयोग कर सकतानी:

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

## प्लेबैक

असिंक्रोनस:

```python
stream.play_async()
while stream.is_playing():
    time.sleep(0.1)
```

सिंक्रोनस:

```python
stream.play()
```

## लाइब्रेरी के परीक्षण

परीक्षण-सबडायरेक्टरी में कई तरह के स्क्रिप्ट बा जे रउआँ के RealtimeTTS लाइब्रेरी के क्षमता के समझे आ जांचे में मदद करेला।

ध्यान दीं कि अधिकतर परीक्षण "पुरानी" OpenAI API (<1.0.0) पर आधारित बा। नया OpenAI API के उपयोग `openai_1.0_test.py` में देखावल गइल बा।

- **simple_test.py**
    - **विवरण**: लाइब्रेरी के सबसे आसान उपयोग के "Hello World" प्रदर्शन।
  
- **complex_test.py**
    - **विवरण**: लाइब्रेरी के अधिकतम कार्यक्षमता के विस्तार।

- **coqui_test.py**
    - **विवरण**: स्थानीय Coqui TTS-इंजन के परीक्षण।

- **translator.py**
    - **आवश्यकता**: `pip install openai realtimestt` कमांड चलाईं।
    - **विवरण**: रियल टाइम में छह अलग-अलग भाषा में अनुवाद।

- **openai_voice_interface.py**
    - **आवश्यकता**: `pip install openai realtimestt` कमांड चलाईं।
    - **विवरण**: एक्टिवेशन-वर्ड के जरिए आवाज आधारित इंटरफेस के साथ OpenAI API के उपयोग।

- **advanced_talk.py**
    - **आवश्यकता**: `pip install openai keyboard realtimestt` कमांड चलाईं।
    - **विवरण**: AI बातचीत शुरू करे से पहले TTS-इंजन आ आवाज के चयन।

- **minimalistic_talkbot.py**
    - **आवश्यकता**: `pip install openai realtimestt` कमांड चलाईं।
    - **विवरण**: 20 लाइन कोड में एगो साधारण टॉकबोट।

- **simple_llm_test.py**
    - **आवश्यकता**: `pip install openai` कमांड चलाईं।
    - **विवरण**: लाइब्रेरी के LLMs के साथ एकीकरण के सरल उदाहरण।

- **test_callbacks.py**
    - **आवश्यकता**: `pip install openai` कमांड चलाईं।
    - **विवरण**: वास्तविक समय में प्रतिक्रिया मापे आ देखावे के।

## रुकल, फेरु से शुरू आ रोकल

ऑडियो स्ट्रीम के रोकल:

```python
stream.pause()
```

रुकल स्ट्रीम के फेरु से शुरू करीं:

```python
stream.resume()
```

स्ट्रीम के तुरंत रोकल:

```python
stream.stop()
```

## आवश्यकताएं

- **Python संस्करण**:
  - **आवश्यक**: Python >= 3.9, < 3.13
  - **कारण**: लाइब्रेरी Coqui के GitHub लाइब्रेरी "TTS" पर निर्भर बा, जे एह संस्करण सीमा के समर्थन करेला।

- **PyAudio**: ऑडियो आउटपुट स्ट्रीम बनावे खातिर

- **stream2sentence**: आवे वाला टेक्स्ट स्ट्रीम के वाक्य में बदलल खातिर

- **pyttsx3**: सिस्टम Text-to-Speech कन्वर्शन इंजन

- **pydub**: ऑडियो चंक फॉर्मेट में बदलल खातिर

- **azure-cognitiveservices-speech**: Azure Text-to-Speech कन्वर्शन इंजन

- **elevenlabs**: Elevenlabs Text-to-Speech कन्वर्शन इंजन

- **coqui-TTS**: उच्च गुणवत्ता वाली स्थानीय न्यूरल TTS खातिर Coqui के XTTS Text-to-Speech लाइब्रेरी

  [Idiap Research Institute](https://github.com/idiap) के धन्यवाद बा उनके [Coqui TTS के Fork](https://github.com/idiap/coqui-ai-TTS) के बनवले खातिर।

- **openai**: OpenAI TTS API के साथ बातचीत खातिर

- **gtts**: Google Translate Text-to-Speech कन्वर्शन