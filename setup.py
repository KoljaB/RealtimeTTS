current_version = "0.5.6"

import setuptools

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
- **minimal**: Core package only (for custom engine development)

You can install multiple engines by separating them with commas. For example:

    pip install realtimetts[azure,elevenlabs,openai]

""" + long_description

# Read requirements.txt and parse it
def parse_requirements(filename):
    requirements = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                package = line.split("==")[0]
                requirements[package] = line
    return requirements


requirements = parse_requirements("requirements.txt")

print("Requirements:", requirements)

# Define all requirements
all_requirements = list(requirements.values())

# Define base requirements
base_requirements = [
    requirements["stream2sentence"],
    requirements["pydub"],
    requirements["pyaudio"],
    requirements["resampy"],
]

# Define subsets of requirements for each engine
extras_require = {
    "minimal": base_requirements,
    "all": base_requirements
    + [requirements["pyttsx3"]]
    + [requirements["azure-cognitiveservices-speech"]]
    + [requirements["elevenlabs"]]
    + [requirements["openai"]]
    + [requirements["gtts"]]
    + [requirements["coqui_tts"]]
    + [requirements["edge-tts"]]
    + [requirements["kokoro"]],    
    "system": base_requirements + [requirements["pyttsx3"]],
    "azure": base_requirements + [requirements["azure-cognitiveservices-speech"]],
    "elevenlabs": base_requirements + [requirements["elevenlabs"]],
    "openai": base_requirements + [requirements["openai"]],
    "gtts": base_requirements + [requirements["gtts"]],
    "coqui": base_requirements + [requirements["coqui_tts"]],
    "edge": base_requirements + [requirements["edge-tts"]],
    "kokoro": base_requirements + [requirements["kokoro"]],
    "orpheus": base_requirements + [requirements["snac"]],
    "jp": base_requirements + ["mecab-python3==1.0.6", "unidic-lite==1.0.8", "cutlet", "fugashi==1.4.0", "jaconv==0.4.0", "mojimoji==0.0.13", "pyopenjtalk==0.4.0"],
    "zh": base_requirements + ["pypinyin==0.53.0", "ordered_set==4.1.0", "jieba==0.42.1", "cn2an==0.5.23"],
    "ko": base_requirements + ["hangul_romanize"],
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
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9, <3.13",
    install_requires=base_requirements,
    extras_require=extras_require,
    package_data={"RealtimeTTS": ["engines/*.json"]},
    include_package_data=True,
    keywords="real-time, text-to-speech, TTS, streaming, audio, voice, synthesis, sentence-segmentation, low-latency, character-streaming, dynamic feedback, audio-output, text-input, TTS-engine, audio-playback, stream-player, sentence-fragment, audio-feedback, interactive, python",
)
