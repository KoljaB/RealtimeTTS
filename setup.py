"""
Setup script for RealtimeTTS-ModelsLab

A ModelsLab TTS engine for the RealtimeTTS library.

Installation:
    pip install realtimetts-modelslab

Or from source:
    pip install .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="realtimetts-modelslab",
    version="1.0.0",
    author="Patcher",
    author_email="patcher@modelslab.com",
    description="ModelsLab TTS engine for RealtimeTTS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/modelslab/realtimetts-modelslab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Sound :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "RealtimeTTS>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    keywords="text-to-speech tts speech synthesis modelslab ai",
    project_urls={
        "Bug Reports": "https://github.com/modelslab/realtimetts-modelslab/issues",
        "Source": "https://github.com/modelslab/realtimetts-modelslab",
        "Documentation": "https://docs.modelslab.com/voice-cloning/text-to-speech",
    },
)
