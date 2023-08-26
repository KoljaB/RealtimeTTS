import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Read requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="RealTimeTTS", 
    version="0.1.0",
    author="Kolja Beigel",
    author_email="kolja.beigel@web.de",
    description="A real-time Text-to-Speech streaming tool that efficiently converts streamed input text into audio with low latency. Ideal for applications requiring instant and dynamic audio feedback.",
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
    keywords='realtime, text streaming, stream, sentence, sentence detection, sentence generation, tts, speech synthesis, nltk, text analysis, audio processing, boundary detection, sentence boundary detection'
)