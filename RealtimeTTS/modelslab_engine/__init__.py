"""
ModelsLab TTS Engine for RealtimeTTS

A text-to-speech engine that uses ModelsLab's TTS API.
Provides high-quality, low-latency text-to-speech synthesis.

Quick Start:
    from RealtimeTTS import TextToAudioStream
    from RealtimeTTS.modelslab_engine import ModelsLabEngine

    engine = ModelsLabEngine(api_key="YOUR_API_KEY")
    stream = TextToAudioStream(engine)
    stream.play("Hello, world!")

For more options:
    engine = ModelsLabEngine(
        api_key="YOUR_API_KEY",
        voice="male1",           # male1, male2, male3, female1, female2, female3
        language="english",       # english, german, french, spanish, etc.
        speed=1.0,               # Speech speed (0.5 - 2.0)
        emotion=False,            # Enable emotion
        temp=False               # Enable temperature control
    )

Environment Variables:
    MODELSLAB_API_KEY: Your ModelsLab API key
    MODELSLAB_VOICE: Default voice
    MODELSLAB_LANGUAGE: Default language
    MODELSLAB_SPEED: Default speed

API Documentation:
    https://docs.modelslab.com/voice-cloning/text-to-speech
"""

from RealtimeTTS.modelslab_engine import ModelsLabEngine, create_engine_from_env

__all__ = ["ModelsLabEngine", "create_engine_from_env"]
__version__ = "1.0.0"
