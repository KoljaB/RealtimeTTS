# RealtimeTTS

[EN](../en/index.md) | [FR](../fr/index.md) | [ES](../es/index.md) | [DE](../de/index.md) | [IT](../it/index.md) | [ZH](../zh/index.md) | [HN](../hn/index.md) | [BH](../bh/index.md)


> **ध्यान दें:** अब `pip install realtimetts` के बेसिक इंस्टॉलेशन के बजाय `pip install realtimetts[all]` के इस्तेमाल करे के सलाह दिहल जा रहल बा।

RealtimeTTS लाइब्रेरी कई तरह के निर्भरता के साथ इंस्टॉलेशन के विकल्प देत बा, ताकि रउआं अपने जरूरत के हिसाब से एकरा के इंस्टॉल क सकेनी। नीचे इंस्टॉलेशन के कुछ विकल्प दिहल गइल बा:

### पूरा इंस्टॉलेशन

सभी TTS इंजिनन के समर्थन के साथ RealtimeTTS स्थापित करे खातिर:

```
pip install -U realtimetts[all]
```

### कस्टम इंस्टॉलेशन

RealtimeTTS में कम से कम लाइब्रेरी के साथ कस्टम इंस्टॉलेशन के सुविधा बा। उपलब्ध विकल्प बा:

- **all**: पूरा इंस्टॉलेशन, सभी इंजिनन के साथ।
- **system**: सिस्टम-विशिष्ट TTS सुविधा शामिल करेला (जइसे pyttsx3)।
- **azure**: Azure Cognitive Services Speech के समर्थन जोड़ेला।
- **elevenlabs**: ElevenLabs API से एकीकरण करेला।
- **openai**: OpenAI वॉइस सेवाओं के समर्थन खातिर।
- **gtts**: Google Text-to-Speech समर्थन।
- **coqui**: Coqui TTS इंजन स्थापित करेला।
- **minimal**: सिर्फ बेस आवश्यकताएं स्थापित करेला, बिना कवनो इंजिन के (अगर रउआं खुद के इंजिन विकसित करे चाहत बानी त ई विकल्प काम के बा)।

उदाहरण के तौर पर, अगर केवल स्थानीय न्यूरल Coqui TTS के उपयोग खातिर RealtimeTTS स्थापित करे के बा त:

```
pip install realtimetts[coqui]
```

अगर रउआं सिर्फ Azure Cognitive Services Speech, ElevenLabs, अउरी OpenAI के समर्थन चाहत बानी त:

```
pip install realtimetts[azure,elevenlabs,openai]
```

### वर्चुअल एनवायरनमेंट इंस्टॉलेशन

अगर रउआं एक वर्चुअल एनवायरनमेंट में पूरा इंस्टॉलेशन करे चाहत बानी, त नीचे के स्टेप फॉलो करीं:

```
python -m venv env_realtimetts
env_realtimetts\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install -U realtimetts[all]
```

CUDA इंस्टॉलेशन से जुड़ल जानकारी खातिर [CUDA इंस्टॉलेशन](#cuda-installation) पर देखल जाव।

## इंजिन आवश्यकताएं

RealtimeTTS में अलग-अलग इंजिनन खातिर अलग-अलग आवश्यकता बा। अपने हिसाब से ई आवश्यकता के पूरा करे के ध्यान दीं।

### SystemEngine
`SystemEngine` अपने सिस्टम के अइसेही मौजूद TTS सुविधा के साथ काम करेला। कवनो अतिरिक्त सेटअप के जरूरत नइखे।

### GTTSEngine
`GTTSEngine` Google Translate के टेक्स्ट-टू-स्पीच API के इस्तेमाल से काम करेला। कवनो अतिरिक्त सेटअप के जरूरत नइखे।

### OpenAIEngine
`OpenAIEngine` के उपयोग खातिर:
- पर्यावरण वेरिएबल OPENAI_API_KEY सेट करीं
- ffmpeg स्थापित करीं (देखल जाव [CUDA इंस्टॉलेशन](#cuda-installation) बिंदु 3)

### AzureEngine
`AzureEngine` के इस्तेमाल खातिर रउआं के चाहीं:
- Microsoft Azure Text-to-Speech API कुंजी (AzureEngine में "speech_key" पैरामीटर के जरिए या पर्यावरण वेरिएबल AZURE_SPEECH_KEY में)
- Microsoft Azure सेवा क्षेत्र।

ई क्रेडेंशियल सेटअप के दौरान सही से कन्फ़िगर होखल चाहिए।

### ElevenlabsEngine
`ElevenlabsEngine` के उपयोग खातिर रउआं के चाहीं:
- Elevenlabs API कुंजी (ElevenlabsEngine में "api_key" पैरामीटर के जरिए या पर्यावरण वेरिएबल ELEVENLABS_API_KEY में)
- रउआं के सिस्टम पर `mpv` स्थापित होखल चाहिए (mpeg ऑडियो स्ट्रीमिंग खातिर आवश्यक बा, काहेंकि Elevenlabs सिर्फ mpeg प्रदान करेला)।

🔹 **`mpv` स्थापित करना**:
  - **macOS**:
    ```
    brew install mpv
    ```

  - **Linux अउरी Windows**: इंस्टॉलेशन के निर्देश खातिर [mpv.io](https://mpv.io/) पर देखल जाव।

### CoquiEngine

उच्च गुणवत्ता के स्थानीय, न्यूरल TTS सेवा देवे ला जे वॉइस क्लोनिंग भी कर सकेला।

पहली बार में एगो न्यूरल TTS मॉडल डाउनलोड करेला। GPU सिंथेसिस के इस्तेमाल करत रियल-टाइम के हिसाब से काफी तेज़ हो सकेला। करीब 4-5 GB VRAM के जरूरत होला।

- वॉइस क्लोनिंग खातिर CoquiEngine के "voice" पैरामीटर में एगो वेव फाइल के नाम दर्ज करीं जेकरा में स्रोत आवाज़ होखे।
- वॉइस क्लोनिंग खातिर 22050 Hz मोनो 16-बिट WAV फाइल के साथ लगभग 5-30 सेकंड के ऑडियो फाइल सबसे बढ़िया परिणाम देला।

### CUDA इंस्टॉलेशन

जिनका लगे NVIDIA GPU बा आ उ बेहतर प्रदर्शन चाहत बानी, ओह लोग खातिर ई स्टेप अनुशंसित बा।

> **ध्यान दें:** *रउआं के NVIDIA GPU CUDA सपोर्ट करेला कि नइखे, ई देखे खातिर [आधिकारिक CUDA GPUs सूची](https://developer.nvidia.com/cuda-gpus) पर चेक करीं।*

CUDA समर्थन के साथ torch इस्तेमाल करे खातिर, ई स्टेप फॉलो करीं:

1. **NVIDIA CUDA टूलकिट स्थापित करीं**:
    उदाहरण खातिर, टूलकिट 12.X स्थापित करे खातिर:
    - [NVIDIA CUDA डाउनलोड](https://developer.nvidia.com/cuda-downloads) पर जाएं।
    - अपने ऑपरेटिंग सिस्टम, सिस्टम आर्किटेक्चर, अउरी ओएस संस्करण के चयन करीं।
    - सॉफ़्टवेयर डाउनलोड आ इंस्टॉल करीं।

2. **NVIDIA cuDNN स्थापित करीं**:

    उदाहरण खातिर, CUDA 11.x खातिर cuDNN 8.7.0 स्थापित करे खातिर:
    - [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) पर जाएं।
    - "Download cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x" पर क्लिक करीं।
    - सॉफ़्टवेयर डाउनलोड आ इंस्टॉल करीं।

3. **ffmpeg स्थापित करीं**:

    रउआं अपने OS खातिर ffmpeg वेबसाइट से इंस्टॉलर डाउनलोड कर सकेनीं: [ffmpeg Website](https://ffmpeg.org/download.html)।

4. **CUDA समर्थन के साथ PyTorch स्थापित करीं**:

    अपने सिस्टम अउरी जरूरत के अनुसार PyTorch संस्करण के CUDA समर्थन के साथ अपग्रेड करे खातिर:

    - **CUDA 11.8 खातिर**:

        ```
        pip install torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
        ```

    - **CUDA 12.X खातिर**:

        ```
        pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
        ```

5. **संगतता समस्या ठीक करे के उपाय**:
    अगर रउआं लाइब्रेरी संगतता के समस्या देखत बानी त ई लाइब्रेरी संस्करणन के फिक्स करे के कोशिश करीं:

    ``` 
    pip install networkx==2.8.8
    pip install typing_extensions==4.8.0
    pip install fsspec==2023.6.0
    pip install imageio==2.31.6
    pip install numpy==1.24.3
    pip install requests==2.31.0
    ```