
> **ملاحظة:** التثبيت الأساسي باستخدام `pip install realtimetts` لم يعد موصى به، استخدم `pip install realtimetts[all]` بدلاً من ذلك.

توفر مكتبة RealtimeTTS خيارات تثبيت لمختلف التبعيات لحالتك الاستخدامية. إليك الطرق المختلفة التي يمكنك من خلالها تثبيت RealtimeTTS حسب احتياجاتك:

### التثبيت الكامل

لتثبيت RealtimeTTS مع دعم لجميع محركات تحويل النص إلى كلام:

```
pip install -U realtimetts[all]
```

### التثبيت المخصص

يسمح RealtimeTTS بالتثبيت المخصص مع الحد الأدنى من تثبيت المكتبات. إليك الخيارات المتاحة:
- **الكل**: التثبيت الكامل مع دعم كل المحركات.
- **النظام**: يشمل قدرات تحويل النص إلى كلام الخاصة بالنظام (e.g., pyttsx3).
- **azure**: يضيف دعم خدمات Azure Cognitive Services Speech.
- **elevenlabs**: يتضمن التكامل مع واجهة برمجة تطبيقات ElevenLabs.
- **openai**: لخدمات الصوت من OpenAI.
- **gtts**: دعم Google Text-to-Speech.
- **coqui**: يقوم بتثبيت محرك Coqui TTS.
- **minimal**: يقوم بتثبيت المتطلبات الأساسية فقط بدون محرك (only needed if you want to develop an own engine)


قل أنك تريد تثبيت RealtimeTTS للاستخدام المحلي فقط مع Coqui TTS العصبي، فعليك استخدام:

```
pip install realtimetts[coqui]
```

على سبيل المثال، إذا كنت ترغب في تثبيت RealtimeTTS مع دعم Azure Cognitive Services Speech و ElevenLabs و OpenAI فقط:

```
pip install realtimetts[azure,elevenlabs,openai]
```

### تثبيت البيئة الافتراضية

بالنسبة لأولئك الذين يرغبون في إجراء تثبيت كامل داخل بيئة افتراضية، اتبعوا هذه الخطوات:

```
python -m venv env_realtimetts
env_realtimetts\Scripts\activate.bat
python.exe -m pip install --upgrade pip
pip install -U realtimetts[all]
```

مزيد من المعلومات حول [تثبيت CUDA](#cuda-installation).

## متطلبات المحرك

تتطلب المحركات المختلفة المدعومة من RealtimeTTS متطلبات فريدة. تأكد من أنك تلبي هذه المتطلبات بناءً على المحرك الذي تختاره.

### محرك النظام
يعمل `SystemEngine` مباشرة مع قدرات تحويل النص إلى كلام المدمجة في نظامك. لا حاجة لأي إعداد إضافي.

### GTTSEngine
يعمل `GTTSEngine` بشكل مباشر باستخدام واجهة برمجة التطبيقات لتحويل النص إلى كلام من Google Translate. لا حاجة لأي إعداد إضافي.

### OpenAIEngine
لاستخدام `OpenAIEngine`:
- تعيين متغير البيئة OPENAI_API_KEY
- تثبيت `realtimetts[openai]` (يتضمن PyAV لتشغيل MP3).

### AzureEngine
لاستخدام `AzureEngine`، ستحتاج إلى:
- مفتاح واجهة برمجة تطبيقات تحويل النص إلى كلام من Microsoft Azure (المقدم عبر معامل منشئ AzureEngine "speech_key" أو في متغير البيئة AZURE_SPEECH_KEY)
- منطقة خدمة Microsoft Azure.

تأكد من أن لديك هذه البيانات متاحة ومهيأة بشكل صحيح عند تهيئة `AzureEngine`.

### محرك Elevenlabs
بالنسبة لـ `ElevenlabsEngine`، تحتاج إلى:
- مفتاح واجهة برمجة تطبيقات Elevenlabs (المقدم عبر معلمة منشئ ElevenlabsEngine "api_key" أو في متغير البيئة ELEVENLABS_API_KEY)
- تم تثبيت `mpv` على نظامك (essential for streaming mpeg audio, Elevenlabs only delivers mpeg).

  🔹 **تثبيت `mpv`:**
  - **macOS**:
    ```
    brew install mpv
```

  - **لينكس وويندوز**: قم بزيارة [mpv.io](https://mpv.io/) للحصول على تعليمات التثبيت.

### CoquiEngine

يوفر تحويل النص إلى كلام العصبي المحلي عالي الجودة مع استنساخ الصوت.

يقوم بتحميل نموذج TTS العصبي أولاً. في معظم الحالات، سيكون سريعًا بما يكفي للتشغيل في الوقت الحقيقي باستخدام تركيب GPU. يحتاج إلى حوالي 4-5 جيجابايت من ذاكرة الوصول العشوائي للرسوميات.

- لاستنساخ الصوت، قدم اسم ملف wav يحتوي على الصوت المصدر كمعامل "voice" إلى مُنشئ CoquiEngine
- يعمل استنساخ الصوت بشكل أفضل مع ملف WAV أحادي 16 بت بتردد 22050 هرتز يحتوي على عينة قصيرة (~5-30 ثانية)

في معظم الأنظمة، ستكون هناك حاجة لدعم وحدة معالجة الرسوميات (GPU) لتشغيلها بسرعة كافية في الوقت الحقيقي، وإلا ستواجه تلعثماً.


### تثبيت CUDA

تُوصى هذه الخطوات لمن يحتاجون إلى **أداء أفضل** ولديهم وحدة معالجة رسومات NVIDIA متوافقة.

> **ملاحظة**: *للتحقق مما إذا كانت بطاقة NVIDIA الرسومية الخاصة بك تدعم CUDA، قم بزيارة [قائمة بطاقات CUDA الرسمية](https://developer.nvidia.com/cuda-gpus).*

لاستخدام Torch مع الدعم عبر CUDA، يرجى اتباع الخطوات التالية:

> **ملاحظة**: *قد لا تحتاج إصدارات PyTorch الأحدث [إلى](https://stackoverflow.com/a/77069523) (غير مؤكدة) إلى تثبيت Toolkit (وربما cuDNN) بعد الآن.*

1. **تثبيت NVIDIA CUDA Toolkit**:
    على سبيل المثال، لتثبيت Toolkit 12.X، يرجى
    - زيارة [تنزيلات NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads).
    - اختر نظام التشغيل الخاص بك، بنية النظام، وإصدار النظام.
    - قم بتنزيل وتثبيت البرنامج.

    أو لتثبيت Toolkit 11.8، يرجى
    - زيارة [أرشيف NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive).
    - اختر نظام التشغيل الخاص بك، بنية النظام، وإصدار نظام التشغيل.
    - قم بتنزيل وتثبيت البرنامج.

٢. **تثبيت NVIDIA cuDNN**:

    على سبيل المثال، لتثبيت cuDNN 8.7.0 لـ CUDA 11.x يرجى
    - زيارة [أرشيف NVIDIA cuDNN](https://developer.nvidia.com/rdp/cudnn-archive).
    - انقر على "تحميل cuDNN v8.7.0 (28 نوفمبر 2022)، لـ CUDA 11.x".
    - قم بتنزيل وتثبيت البرنامج.

٣. **تثبيت ffmpeg**:

    يمكنك تنزيل مثبت لنظام التشغيل الخاص بك من [موقع ffmpeg](https://ffmpeg.org/download.html).

    أو استخدم مدير حزم:

    - **على أوبونتو أو ديبيان**:
        ```
        sudo apt update && sudo apt install ffmpeg
        ```

    - **على أرتش لينكس**:
        ```
        sudo pacman -S ffmpeg
        ```

    - **على نظام MacOS باستخدام Homebrew** ([https://brew.sh/](https://brew.sh/)):
        ```
        brew install ffmpeg
        ```

    - **على نظام ويندوز باستخدام Chocolatey** ([https://chocolatey.org/](https://chocolatey.org/)):
        ```
        choco install ffmpeg
```

    - **على نظام ويندوز باستخدام سكوب** ([https://scoop.sh/](https://scoop.sh/)):
        ```
        سكووب تثبيت ffmpeg
```

٤. **تثبيت PyTorch مع دعم CUDA**:

    لترقية تثبيت PyTorch الخاص بك لتمكين دعم GPU باستخدام CUDA، اتبع هذه التعليمات بناءً على إصدار CUDA الخاص بك. هذا مفيد إذا كنت ترغب في تحسين أداء RealtimeSTT بقدرات CUDA.

    - **لـ CUDA 11.8:**

        لتحديث PyTorch و Torchaudio لدعم CUDA 11.8، استخدم الأوامر التالية:

        ```
        pip install torch==2.3.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```
        النص للترجمة: ```

    - **لـ CUDA 12.X:**


        لتحديث PyTorch و Torchaudio لدعم CUDA 12.X، نفذ ما يلي:

        ```
        pip install torch==2.3.1+cu121 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
        النص للترجمة: ```

    استبدل `2.3.1` بالإصدار من PyTorch الذي يتناسب مع نظامك ومتطلباتك.

٥. **إصلاح لحل مشاكل التوافق**:
    إذا واجهت مشاكل في توافق المكتبات، حاول ضبط هذه المكتبات على إصدارات ثابتة:

  النص للترجمة: ``` 

    pip install networkx==2.8.8
    
    pip install typing_extensions==4.8.0
    
    pip install fsspec==2023.6.0
    
    pip install imageio==2.31.6
    
    pip install networkx==2.8.8
    
    pip install numpy==1.24.3
    
    pip install requests==2.31.0
  ```