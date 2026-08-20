import os
import re
from pathlib import Path

import setuptools

_ROOT = Path(__file__).resolve().parent
_version_text = (Path(__file__).resolve().parent / "RealtimeTTS" / "_version.py").read_text(
    encoding="utf-8"
)
_version_match = re.search(r'^__version__ = "([^"]+)"$', _version_text, re.MULTILINE)
if _version_match is None:
    raise RuntimeError("Could not determine RealtimeTTS package version")
current_version = _version_match.group(1)

# Read the contents of README.md
with (_ROOT / "README.md").open("r", encoding="utf-8") as fh:
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
- **qwen**: Native qwentts.cpp Qwen3 TTS integration
- **qwen-server**: OpenAI-compatible native Qwen3 TTS HTTP server
- **omnivoice**: Omnivoice TTS integration
- **luxtts**: LuxTTS integration
- **chatterbox**: Chatterbox Turbo integration
- **inflect**: Inflect-Micro-v2 local TTS integration
- **inflect-pytorch**: Inflect-Micro-v2 PyTorch backend only
- **inflect-onnx**: Inflect-Micro-v2 ONNX backend only
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
- **nltk**: Default NLTK plus rule-based sentence tokenizer (included by default)
- **stanza**: Add the optional Stanza sentence tokenizer
- **minimal**: Core package only (for custom engine development)

You can install multiple engines by separating them with commas. For example:

    pip install realtimetts[azure,elevenlabs,openai]

""" + long_description

# Read requirements.txt and parse it
def parse_requirements(filename):
    requirements = {}
    with (_ROOT / filename).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Split by any version operator: =, >, <, ~, or !
                package = re.split(r'[=><~!]', line)[0].strip().split("[", 1)[0]
                requirements[package] = line
    return requirements


requirements = parse_requirements("requirements.txt")

# Define all requirements
all_requirements = list(requirements.values())

# Define base requirements (using .get() to prevent KeyErrors if missing from requirements.txt)
base_requirements =[
    requirements.get("stream2sentence", "stream2sentence[nltk]>=1.0.3"),
    requirements.get("pydub", "pydub"),
    requirements.get(
        "audioop-lts",
        'audioop-lts>=0.2.2; python_version >= "3.13"',
    ),
    requirements.get("resampy", "resampy"),
]
stanza_tokenizer_requirements = ["stream2sentence[stanza]>=1.0.3"]
pyaudio_requirements = [requirements.get("pyaudio", "pyaudio>=0.2.14")]
standard_requirements = base_requirements + pyaudio_requirements

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
# Do not look up qwentts-cpp-python through the name-keyed requirements mapping:
# the platform pins intentionally share one package name, so parse_requirements
# would otherwise keep only the last variant.
qwen_native_requirements = [
    'qwentts-cpp-python[cuda12]==0.4.0.dev1; sys_platform == "win32"',
    'qwentts-cpp-python[cuda12]==0.4.0.dev0; sys_platform == "linux"',
]
qwen_common_requirements = qwen_native_requirements + [
    requirements.get("numpy", "numpy"),
    requirements.get("soundfile", "soundfile>=0.13.1"),
]
qwen_requirements = qwen_common_requirements + pyaudio_requirements
qwen_server_requirements = [
    requirements.get("fastapi", "fastapi>=0.115,<1"),
    requirements.get("uvicorn", "uvicorn>=0.34,<1"),
    requirements.get("fasttext-predict", "fasttext-predict>=0.9.2.4"),
]
orpheus_requirements = [requirements.get("snac", "snac")]
omnivoice_requirements = [requirements.get("omnivoice", "omnivoice")]
chatterbox_requirements = [requirements.get("chatterbox-tts", "chatterbox-tts")]
inflect_common_requirements = [
    "numpy>=1.26,<3",
    "soundfile>=0.13",
    "huggingface-hub>=0.36",
    "phonemizer>=3.3",
    "espeakng-loader>=0.2.4",
    "num2words>=0.5.14",
    "Unidecode>=1.3.8",
]
inflect_pytorch_requirements = inflect_common_requirements + [
    "torch>=2.6",
    "scipy>=1.13",
]
inflect_onnx_requirements = inflect_common_requirements + [
    "onnxruntime>=1.18,<2",
]
inflect_requirements = inflect_common_requirements + [
    "torch>=2.6",
    "scipy>=1.13",
    "onnxruntime>=1.18,<2",
]
sopro_requirements = [requirements.get("sopro", "sopro")]
soprano_requirements = [requirements.get("soprano-tts", "soprano-tts")]
neutts_requirements = [requirements.get("neutts", "neutts")]
pockettts_requirements = [
    requirements.get("pocket-tts", "pocket-tts"),
    requirements.get("torch", "torch"),
]
pockettts_gpu_requirements = [
    requirements.get("torch", "torch"),
    requirements.get("scipy", "scipy"),
    requirements.get("safetensors", "safetensors"),
    requirements.get("huggingface-hub", "huggingface-hub"),
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
    + qwen_server_requirements
    + orpheus_requirements
    + omnivoice_requirements
    + chatterbox_requirements
    + inflect_requirements
    + sopro_requirements
    + soprano_requirements
    + neutts_requirements
    + pockettts_requirements
    + pockettts_gpu_requirements
    + zipvoice_requirements
    + luxtts_requirements
    + styletts_requirements
    + parler_requirements
    + moss_requirements
)

extras_require = {
    "minimal": base_requirements,
    "playback": standard_requirements,
    "all": standard_requirements + stanza_tokenizer_requirements + all_engine_requirements,
    "nltk": base_requirements,
    "stanza": base_requirements + stanza_tokenizer_requirements,
    "system": standard_requirements + system_requirements,
    "azure": standard_requirements + azure_requirements,
    "elevenlabs": standard_requirements + elevenlabs_requirements,
    "openai": standard_requirements + openai_requirements,
    "gtts": standard_requirements + gtts_requirements,
    "coqui": standard_requirements + coqui_requirements,
    "edge": standard_requirements + edge_requirements,
    "kokoro": standard_requirements + kokoro_requirements,
    "camb": standard_requirements + camb_requirements,
    "minimax": standard_requirements + requests_requirements,
    "modelslab": standard_requirements + requests_requirements,
    "cartesia": standard_requirements + cartesia_requirements,
    "typecast": standard_requirements + typecast_requirements,
    "orpheus": standard_requirements + orpheus_requirements,
    "omnivoice": standard_requirements + omnivoice_requirements,
    "luxtts": standard_requirements + luxtts_requirements,
    "zipvoice": standard_requirements + zipvoice_requirements,
    "chatterbox": standard_requirements + chatterbox_requirements,
    "inflect": standard_requirements + inflect_requirements,
    "inflect-pytorch": standard_requirements + inflect_pytorch_requirements,
    "inflect-onnx": standard_requirements + inflect_onnx_requirements,
    "sopro": standard_requirements + sopro_requirements,
    "soprano": standard_requirements + soprano_requirements,
    "neutts": standard_requirements + neutts_requirements,
    "neutts-gguf": standard_requirements + ["neutts[llama,onnx]"],
    "pockettts": standard_requirements + pockettts_requirements,
    "pocket": standard_requirements + pockettts_requirements,
    "pockettts-gpu": standard_requirements + pockettts_gpu_requirements,
    "pocket-gpu": standard_requirements + pockettts_gpu_requirements,
    "styletts": standard_requirements + styletts_requirements,
    "style": standard_requirements + styletts_requirements,
    "parler": standard_requirements + parler_requirements,
    "moss": standard_requirements + moss_requirements,
    "moss-tts": standard_requirements + moss_requirements,
    "piper": standard_requirements,
    # Qwen uses PyAudio for the in-process playback path. The server extra is
    # headless and therefore intentionally leaves PyAudio out.
    "qwen": base_requirements + qwen_requirements,
    "qwen-server": base_requirements + qwen_common_requirements + qwen_server_requirements,
    "jp": standard_requirements +["mecab-python3>=1.0.12", "unidic-lite>=1.0.8", "cutlet", "fugashi>=1.5.2", "jaconv>=0.5.0", "mojimoji>=0.0.13", "pyopenjtalk>=0.4.1"],
    "zh": standard_requirements +["pypinyin>=0.55.0", "ordered_set>=4.1.0", "jieba>=0.42.1", "cn2an>=0.5.24"],
    "ko": standard_requirements +["hangul_romanize"],
}

os.chdir(_ROOT)

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
    license_files=["LICENSE", "LICENSING_ADDENDUM.md"],
    python_requires=">=3.9, <3.15",
    install_requires=base_requirements,
    extras_require=extras_require,
    package_data={"RealtimeTTS": ["engines/*.json"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "realtimetts-qwen-server=RealtimeTTS.qwen_server:main",
        ],
    },
    keywords="real-time, text-to-speech, TTS, streaming, audio, voice, synthesis, sentence-segmentation, low-latency, character-streaming, dynamic feedback, audio-output, text-input, TTS-engine, audio-playback, stream-player, sentence-fragment, audio-feedback, interactive, python",
)
