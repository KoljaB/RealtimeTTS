# RealtimeTTS

[EN](../en/index.md) | [FR](../fr/index.md) | [ES](../es/index.md) | [DE](../de/index.md) | [IT](../it/index.md) | [ZH](../zh/index.md) | [HI](index.md)

> **नोट:** `pip install realtimetts` का बेसिक इंस्टॉलेशन अब अनुशंसित नहीं है, इसके बजाय `pip install realtimetts[all]` का उपयोग करें।

RealtimeTTS लाइब्रेरी विभिन्न निर्भरताओं के साथ इंस्टॉलेशन के विकल्प प्रदान करती है, ताकि आप अपने उपयोग के अनुसार इसे स्थापित कर सकें। यहाँ इंस्टॉलेशन के विभिन्न विकल्प दिए गए हैं:

### पूर्ण स्थापना

सभी TTS इंजनों के समर्थन के साथ RealtimeTTS स्थापित करने के लिए:

```
pip install -U realtimetts[all]
```

### कस्टम इंस्टॉलेशन

RealtimeTTS में न्यूनतम लाइब्रेरी इंस्टॉलेशन के साथ कस्टम इंस्टॉलेशन की सुविधा है। उपलब्ध विकल्प:

- **all**: सभी इंजनों के साथ पूर्ण इंस्टॉलेशन।
- **system**: सिस्टम-विशिष्ट TTS क्षमताएँ शामिल करता है (जैसे, pyttsx3)।
- **azure**: Azure Cognitive Services Speech का समर्थन जोड़ता है।
- **elevenlabs**: ElevenLabs API के साथ एकीकरण।
- **openai**: OpenAI वॉइस सेवाओं के लिए।
- **gtts**: Google Text-to-Speech समर्थन।
- **coqui**: Coqui TTS इंजन स्थापित करता है।
- **minimal**: केवल बेस आवश्यकताओं को स्थापित करता है, बिना किसी इंजन के (यदि आप अपना इंजन विकसित करना चाहते हैं तो इसकी आवश्यकता होती है)।

उदाहरण के लिए, केवल स्थानीय न्यूरल Coqui TTS उपयोग के लिए RealtimeTTS स्थापित करना हो, तो उपयोग करें:

```
pip install realtimetts[coqui]
```

अगर आप केवल Azure Cognitive Services Speech, ElevenLabs, और OpenAI का समर्थन चाहते हैं तो:

```
pip install realtimetts[azure,elevenlabs,openai]
```

### वर्चुअल एनवायरनमेंट इंस्टॉलेशन

यदि आप एक वर्चुअल एनवायरनमेंट में पूर्ण स्थापना करना चाहते हैं, तो ये कदम अपनाएँ:

```
python -m venv env_realtimetts
env_realtimetts\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install -U realtimetts[all]
```

[CUDA इंस्टॉलेशन](#cuda-installation) के बारे में अधिक जानकारी।

## इंजन आवश्यकताएँ

RealtimeTTS द्वारा समर्थित विभिन्न इंजनों की अलग-अलग आवश्यकताएँ हैं। अपनी पसंद के अनुसार इन आवश्यकताओं को पूरा करना सुनिश्चित करें।

### SystemEngine
`SystemEngine` आपके सिस्टम की अंतर्निहित TTS क्षमताओं के साथ स्वतः काम करता है। किसी अतिरिक्त सेटअप की आवश्यकता नहीं है।

### GTTSEngine
`GTTSEngine` Google Translate के टेक्स्ट-टू-स्पीच API का उपयोग करके स्वतः काम करता है। किसी अतिरिक्त सेटअप की आवश्यकता नहीं है।

### OpenAIEngine
`OpenAIEngine` का उपयोग करने के लिए:
- पर्यावरण वेरिएबल OPENAI_API_KEY सेट करें
- `realtimetts[openai]` स्थापित करें (MP3 प्लेबैक के लिए PyAV शामिल है)।

### AzureEngine
`AzureEngine` का उपयोग करने के लिए आपको चाहिए:
- Microsoft Azure Text-to-Speech API कुंजी (AzureEngine में "speech_key" पैरामीटर के माध्यम से या पर्यावरण वेरिएबल AZURE_SPEECH_KEY में)
- Microsoft Azure सेवा क्षेत्र।

इंस्टॉल करते समय ये क्रेडेंशियल उपलब्ध और सही तरीके से कॉन्फ़िगर करना सुनिश्चित करें।

### ElevenlabsEngine
`ElevenlabsEngine` के लिए, आपको चाहिए:
- Elevenlabs API कुंजी (ElevenlabsEngine में "api_key" पैरामीटर के माध्यम से या पर्यावरण वेरिएबल ELEVENLABS_API_KEY में)
- आपके सिस्टम पर `mpv` स्थापित हो (mpeg ऑडियो स्ट्रीमिंग के लिए आवश्यक है, Elevenlabs केवल mpeg प्रदान करता है)।

  🔹 **`mpv` स्थापित करना**:
  - **macOS**:
    ```
    brew install mpv
    ```

  - **Linux और Windows**: इंस्टॉलेशन के निर्देशों के लिए [mpv.io](https://mpv.io/) पर जाएं।

### CoquiEngine

उच्च गुणवत्ता, स्थानीय, न्यूरल TTS प्रदान करता है जिसमें वॉइस-क्लोनिंग भी शामिल है।

पहली बार एक न्यूरल TTS मॉडल डाउनलोड करता है। अधिकतर मामलों में GPU सिंथेसिस का उपयोग करते हुए रीयल-टाइम के लिए पर्याप्त तेज़ होगा। लगभग 4-5 GB VRAM की आवश्यकता होती है।

- वॉइस क्लोन करने के लिए CoquiEngine के "voice" पैरामीटर में एक वेव फ़ाइल का नाम दर्ज करें जिसमें स्रोत वॉइस हो।
- वॉइस क्लोनिंग के लिए 22050 Hz मोनो 16-बिट WAV फाइल के साथ लगभग 5-30 सेकंड की नमूना ऑडियो फ़ाइल सबसे अच्छा परिणाम देती है।

### CUDA इंस्टॉलेशन

वे लोग जिनके पास NVIDIA GPU है और जो **बेहतर प्रदर्शन** चाहते हैं, उनके लिए ये कदम अनुशंसित हैं।

> **नोट:** *अगर आपका NVIDIA GPU CUDA को सपोर्ट करता है तो [आधिकारिक CUDA GPUs सूची](https://developer.nvidia.com/cuda-gpus) पर जाँचें।*

CUDA समर्थन के साथ torch का उपयोग करने के लिए, इन चरणों का पालन करें:

1. **NVIDIA CUDA टूलकिट स्थापित करें**:
    उदाहरण के लिए, टूलकिट 12.X स्थापित करने के लिए:
    - [NVIDIA CUDA डाउनलोड](https://developer.nvidia.com/cuda-downloads) पर जाएँ।
    - अपने ऑपरेटिंग सिस्टम, सिस्टम आर्किटेक्चर, और ओएस संस्करण का चयन करें।
    - सॉफ़्टवेयर डाउनलोड और इंस्टॉल करें।

2. **NVIDIA cuDNN स्थापित करें**:

    उदाहरण के लिए, CUDA 11.x के लिए cuDNN 8.7.0 स्थापित करने के लिए:
    - [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) पर जाएं।
    - "Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x" पर क्लिक करें।
    - सॉफ़्टवेयर डाउनलोड और इंस्टॉल करें।

3. **ffmpeg स्थापित करें**:

    आप अपने OS के लिए ffmpeg वेबसाइट से इंस्टॉलर डाउनलोड कर सकते हैं: [ffmpeg Website](https://ffmpeg.org/download.html)।

4. **CUDA समर्थन के साथ PyTorch स्थापित करें**:

    अपने सिस्टम और आवश्यकताओं के अनुसार PyTorch संस्करण को CUDA समर्थन के साथ अपग्रेड करने के लिए:

    - **CUDA 11.8 के लिए**:

        ```
        pip install torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
        ```

    - **CUDA 12.X के लिए**:

        ```
        pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
        ```

5. **संगतता समस्याओं को हल करने के लिए फिक्स**:
    यदि आप लाइब्रेरी संगतता मुद्दों का सामना करते हैं, तो इन लाइब्रेरी संस्करणों को फिक्स करने का प्रयास करें:

  ``` 
    pip install networkx==2.8.8
    pip install typing_extensions==4.8.0
    pip install fsspec==2023.6.0
    pip install imageio==2.31.6
    pip install numpy==1.24.3
    pip install requests==2.31.0
  ```
