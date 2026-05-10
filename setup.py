current_version = "0.7.0"

import setuptools
import re

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

long_description = """
To install realtimetts, you need to specify the TTS engine(s) you wish to use.

For example, to install all supported engines:

    pip install realtimetts[all]

To install with the Coqui TTS engine:

    pip install realtimetts[coqui]

Available engine options include:

- **all**: Install all supported engines
- **system**: Local system TTS via `pyttsx3`
- **azure**: Azure Speech Services support
- **elevenlabs**: ElevenLabs API integration
- **openai**: OpenAI TTS services
- **gtts**: Google Text-to-Speech
- **edge**: Microsoft Edge TTS
- **coqui**: Coqui TTS engine
- **camb**: CAMB AI MARS TTS
- **minimax**: MiniMax Cloud TTS
- **cartesia**: Cartesia API integration
- **modelslab**: ModelsLab API integration
- **orpheus**: Orpheus TTS support
- **qwen**: Faster Qwen3 TTS integration
- **omnivoice**: Omnivoice TTS integration
- **luxtts**: LuxTTS integration
- **chatterbox**: Chatterbox Turbo integration
- **sopro**: SoproTTS integration
- **soprano**: SopranoTTS integration
- **neutts**: NeuTTS integration
- **zipvoice**: ZipVoice dependency support
- **moss**: MOSS-TTS dependency support
- **pockettts**: PocketTTS integration
- **parler**: Parler TTS integration
- **styletts**: StyleTTS integration
- **piper**: Piper executable engine support
- **typecast**: Typecast API integration
- **minimal**: Core package only (for custom engine development)

You can install multiple engines by separating them with commas. For example:

    pip install realtimetts[azure,elevenlabs,openai]

""" + long_description

# Read requirements.txt and parse it
def parse_requirements(filename):
    requirements = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Split by any version operator: =, >, <, ~, or !
                package = re.split(r'[=><~!]', line)[0].strip()
                requirements[package] = line
    return requirements


requirements = parse_requirements("requirements.txt")

print("Requirements:", requirements)

# Define all requirements
all_requirements = list(requirements.values())

# Define base requirements (using .get() to prevent KeyErrors if missing from requirements.txt)
base_requirements =[
    requirements.get("stream2sentence", "stream2sentence"),
    requirements.get("pydub", "pydub"),
    requirements.get("pyaudio", "pyaudio"),
    requirements.get("resampy", "resampy"),
]

# Define subsets of requirements for each engine safely
system_requirements = [requirements.get("pyttsx3", "pyttsx3")]
azure_requirements = [requirements.get("azure-cognitiveservices-speech", "azure-cognitiveservices-speech")]
elevenlabs_requirements = [requirements.get("elevenlabs", "elevenlabs")]
openai_requirements = [requirements.get("openai", "openai")]
gtts_requirements = [requirements.get("gtts", "gtts")]
coqui_requirements = [requirements.get("coqui_tts", "coqui_tts")]
edge_requirements = [requirements.get("edge-tts", "edge-tts")]
kokoro_requirements = [requirements.get("kokoro", "kokoro")]
camb_requirements = [requirements.get("camb-sdk", "camb-sdk")]
requests_requirements = [requirements.get("requests", "requests")]
cartesia_requirements = [requirements.get("cartesia", "cartesia")]
typecast_requirements = [requirements.get("typecast-python", "typecast-python")]
qwen_requirements = [requirements.get("faster-qwen3-tts", "faster-qwen3-tts")]
orpheus_requirements = [requirements.get("snac", "snac")]
omnivoice_requirements = [requirements.get("omnivoice", "omnivoice")]
chatterbox_requirements = [requirements.get("chatterbox-tts", "chatterbox-tts")]
sopro_requirements = [requirements.get("sopro", "sopro")]
soprano_requirements = [requirements.get("soprano-tts", "soprano-tts")]
neutts_requirements = [requirements.get("neutts", "neutts")]
pockettts_requirements = [
    requirements.get("pocket-tts", "pocket-tts"),
    requirements.get("torch", "torch"),
]
zipvoice_requirements = [
    requirements.get("torch", "torch"),
    requirements.get("torchaudio", "torchaudio"),
    requirements.get("numpy", "numpy"),
    requirements.get("huggingface-hub", "huggingface-hub"),
    requirements.get("safetensors", "safetensors"),
    requirements.get("vocos", "vocos"),
]
luxtts_requirements = [
    requirements.get("cn2an", "cn2an"),
    requirements.get("inflect", "inflect"),
    requirements.get("jieba", "jieba"),
    requirements.get("lhotse", "lhotse"),
    requirements.get("librosa", "librosa"),
    requirements.get("numpy", "numpy"),
    requirements.get("onnxruntime", "onnxruntime"),
    requirements.get("piper_phonemize", "piper_phonemize"),
    requirements.get("pydub", "pydub"),
    requirements.get("pypinyin", "pypinyin"),
    requirements.get("safetensors", "safetensors"),
    "setuptools<81",
    requirements.get("tensorboard", "tensorboard"),
    requirements.get("torch", "torch"),
    requirements.get("torchaudio", "torchaudio"),
    "transformers<=4.57.6",
    requirements.get("vocos", "vocos"),
]
styletts_requirements = [
    requirements.get("torch", "torch"),
    requirements.get("torchaudio", "torchaudio"),
    requirements.get("numpy", "numpy"),
    requirements.get("librosa", "librosa"),
    requirements.get("nltk", "nltk"),
    requirements.get("munch", "munch"),
    requirements.get("PyYAML", "PyYAML"),
    requirements.get("phonemizer", "phonemizer"),
]
parler_requirements = [
    requirements.get("torch", "torch"),
    requirements.get("transformers", "transformers"),
]
moss_requirements = [
    requirements.get("numpy", "numpy"),
    requirements.get("soundfile", "soundfile"),
    requirements.get("torch", "torch"),
    requirements.get("torchaudio", "torchaudio"),
    requirements.get("onnxruntime", "onnxruntime"),
    requirements.get("huggingface-hub", "huggingface-hub"),
    requirements.get("nltk", "nltk"),
]

all_engine_requirements = (
    system_requirements
    + azure_requirements
    + elevenlabs_requirements
    + openai_requirements
    + gtts_requirements
    + coqui_requirements
    + edge_requirements
    + kokoro_requirements
    + camb_requirements
    + requests_requirements
    + cartesia_requirements
    + typecast_requirements
    + qwen_requirements
    + orpheus_requirements
    + omnivoice_requirements
    + chatterbox_requirements
    + sopro_requirements
    + soprano_requirements
    + neutts_requirements
    + pockettts_requirements
    + zipvoice_requirements
    + luxtts_requirements
    + styletts_requirements
    + parler_requirements
    + moss_requirements
)

extras_require = {
    "minimal": base_requirements,
    "all": base_requirements + all_engine_requirements,
    "system": base_requirements + system_requirements,
    "azure": base_requirements + azure_requirements,
    "elevenlabs": base_requirements + elevenlabs_requirements,
    "openai": base_requirements + openai_requirements,
    "gtts": base_requirements + gtts_requirements,
    "coqui": base_requirements + coqui_requirements,
    "edge": base_requirements + edge_requirements,
    "kokoro": base_requirements + kokoro_requirements,
    "camb": base_requirements + camb_requirements,
    "minimax": base_requirements + requests_requirements,
    "modelslab": base_requirements + requests_requirements,
    "cartesia": base_requirements + cartesia_requirements,
    "typecast": base_requirements + typecast_requirements,
    "orpheus": base_requirements + orpheus_requirements,
    "omnivoice": base_requirements + omnivoice_requirements,
    "luxtts": base_requirements + luxtts_requirements,
    "zipvoice": base_requirements + zipvoice_requirements,
    "chatterbox": base_requirements + chatterbox_requirements,
    "sopro": base_requirements + sopro_requirements,
    "soprano": base_requirements + soprano_requirements,
    "neutts": base_requirements + neutts_requirements,
    "neutts-gguf": base_requirements + ["neutts[llama,onnx]"],
    "pockettts": base_requirements + pockettts_requirements,
    "pocket": base_requirements + pockettts_requirements,
    "styletts": base_requirements + styletts_requirements,
    "style": base_requirements + styletts_requirements,
    "parler": base_requirements + parler_requirements,
    "moss": base_requirements + moss_requirements,
    "moss-tts": base_requirements + moss_requirements,
    "piper": base_requirements,
    "qwen": base_requirements + qwen_requirements,
    "jp": base_requirements +["mecab-python3>=1.0.6", "unidic-lite>=1.0.8", "cutlet", "fugashi>=1.4.0", "jaconv>=0.4.0", "mojimoji>=0.0.13", "pyopenjtalk>=0.4.0"],
    "zh": base_requirements +["pypinyin>=0.53.0", "ordered_set>=4.1.0", "jieba>=0.42.1", "cn2an>=0.5.23"],
    "ko": base_requirements +["hangul_romanize"],
}

setuptools.setup(
    name="realtimetts",
    version=current_version,
    author="Kolja Beigel",
    author_email="kolja.beigel@web.de",
    description="Stream text into audio with an easy-to-use, highly configurable library delivering voice output with minimal latency.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KoljaB/realtimetts",
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9, <3.15",
    install_requires=base_requirements,
    extras_require=extras_require,
    package_data={"RealtimeTTS": ["engines/*.json"]},
    include_package_data=True,
    keywords="real-time, text-to-speech, TTS, streaming, audio, voice, synthesis, sentence-segmentation, low-latency, character-streaming, dynamic feedback, audio-output, text-input, TTS-engine, audio-playback, stream-player, sentence-fragment, audio-feedback, interactive, python",
)
