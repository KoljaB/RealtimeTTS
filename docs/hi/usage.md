# उपयोग

## त्वरित प्रारंभ

यहाँ एक बुनियादी उदाहरण दिया गया है:

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

engine = SystemEngine() # अपनी TTS-इंजन के साथ इसे बदलें
stream = TextToAudioStream(engine)
stream.feed("Hello world! How are you today?")
stream.play_async()
```

## टेक्स्ट इनपुट

आप व्यक्तिगत स्ट्रिंग्स इनपुट कर सकते हैं:

```python
stream.feed("Hello, this is a sentence.")
```

या आप वास्तविक समय-स्ट्रीमिंग के लिए जनरेटर और कैरेक्टर-इटरेटर का उपयोग कर सकते हैं:

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

## लाइब्रेरी का परीक्षण

परीक्षण-सबडायरेक्टरी में विभिन्न स्क्रिप्ट्स शामिल हैं, जो आपको RealtimeTTS लाइब्रेरी की क्षमताओं को समझने और आकलन करने में मदद करती हैं।

ध्यान दें कि अधिकांश परीक्षण अभी भी "पुरानी" OpenAI API (<1.0.0) पर आधारित हैं। नई OpenAI API का उपयोग `openai_1.0_test.py` में प्रदर्शित किया गया है।

- **simple_test.py**
    - **विवरण**: सबसे आसान लाइब्रेरी उपयोग का एक "Hello World" जैसा प्रदर्शन।

- **complex_test.py**
    - **विवरण**: लाइब्रेरी की अधिकतम कार्यक्षमता का व्यापक प्रदर्शन।

- **coqui_test.py**
    - **विवरण**: स्थानीय Coqui TTS-इंजन का परीक्षण।

- **translator.py**
    - **आवश्यकताएँ**: `pip install openai realtimestt` कमांड चलाएँ।
    - **विवरण**: वास्तविक समय में छह विभिन्न भाषाओं में अनुवाद।

- **openai_voice_interface.py**
    - **आवश्यकताएँ**: `pip install openai realtimestt` कमांड चलाएँ।
    - **विवरण**: एक्टिवेशन-वर्ड के माध्यम से और आवाज-आधारित इंटरफ़ेस के साथ OpenAI API का उपयोग।

- **advanced_talk.py**
    - **आवश्यकताएँ**: `pip install openai keyboard realtimestt` कमांड चलाएँ।
    - **विवरण**: AI बातचीत शुरू करने से पहले TTS-इंजन और आवाज का चयन।

- **minimalistic_talkbot.py**
    - **आवश्यकताएँ**: `pip install openai realtimestt` कमांड चलाएँ।
    - **विवरण**: 20 कोड लाइनों में एक साधारण टॉकबोट।

- **simple_llm_test.py**
    - **आवश्यकताएँ**: `pip install openai` कमांड चलाएँ।
    - **विवरण**: लाइब्रेरी की LLMs के साथ एकीकृत करने का सरल प्रदर्शन।

- **test_callbacks.py**
    - **आवश्यकताएँ**: `pip install openai` कमांड चलाएँ।
    - **विवरण**: वास्तविक वातावरण में विलंब समय को मापने और प्रतिक्रिया को प्रदर्शित करता है।

## रुकना, पुनः आरंभ करना और रोकना

ऑडियो स्ट्रीम को रोकें:

```python
stream.pause()
```

रुकी हुई स्ट्रीम पुनः प्रारंभ करें:

```python
stream.resume()
```

स्ट्रीम तुरंत रोकें:

```python
stream.stop()
```

## आवश्यकताओं का स्पष्टीकरण

- **Python संस्करण**:
  - **आवश्यक**: Python >= 3.9, < 3.13
  - **कारण**: लाइब्रेरी Coqui की GitHub लाइब्रेरी "TTS" पर निर्भर करती है, जो इस संस्करण सीमा का समर्थन करती है।

- **PyAudio**: ऑडियो आउटपुट स्ट्रीम बनाने के लिए

- **stream2sentence**: इनकमिंग टेक्स्ट स्ट्रीम को वाक्यों में विभाजित करने के लिए

- **pyttsx3**: सिस्टम Text-to-Speech कन्वर्शन इंजन

- **pydub**: ऑडियो चंक फॉर्मेट्स में कन्वर्शन के लिए

- **azure-cognitiveservices-speech**: Azure Text-to-Speech कन्वर्शन इंजन

- **elevenlabs**: Elevenlabs Text-to-Speech कन्वर्शन इंजन

- **coqui-TTS**: उच्च गुणवत्ता वाली स्थानीय न्यूरल TTS के लिए Coqui का XTTS Text-to-Speech लाइब्रेरी

  [Idiap Research Institute](https://github.com/idiap) को उनके [Coqui TTS का Fork](https://github.com/idiap/coqui-ai-TTS) बनाए रखने के लिए धन्यवाद।

- **openai**: OpenAI TTS API के साथ बातचीत के लिए

- **gtts**: Google Translate Text-to-Speech कन्वर्शन