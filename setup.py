import setuptools
import os

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()


# Read requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
## detect if os is debian or ubuntu
if os.path.exists("/etc/debian_version") or os.path.exists("/etc/lsb-release"):
    ## install with apt install python3-pyaudio
    os.system("sudo apt install libportaudio2 python3-pyaudio")
    #http://http.us.debian.org/debian/pool/main/p/python-pyaudio/python3-pyaudio_0.2.13-1+b2_amd64.deb
    ## curl and install pyaudio from url
    os.system("curl -O http://http.us.debian.org/debian/pool/main/p/python-pyaudio/python3-pyaudio_0.2.13-1+b2_amd64.deb")
    os.system("sudo dpkg -i python3-pyaudio_0.2.13-1+b2_amd64.deb")
    os.system("rm python3-pyaudio_0.2.13-1+b2_amd64.deb")
    ## remove pyaudio from requirements.txt
    requirements.remove("PyAudio==0.2.14")

setuptools.setup(
    name="RealTimeTTS", 
    version="0.3.41",
    author="Kolja Beigel",
    author_email="kolja.beigel@web.de",
    description="*Stream text into audio with an easy-to-use, highly configurable library delivering voice output with minimal latency.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KoljaB/RealTimeTTS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements,
    package_data={'RealtimeTTS': ['engines/*.json']},
    include_package_data=True,    
    keywords='real-time, text-to-speech, TTS, streaming, audio, voice, synthesis, sentence-segmentation, low-latency, character-streaming, dynamic feedback, audio-output, text-input, TTS-engine, audio-playback, stream-player, sentence-fragment, audio-feedback, interactive, python'
)
