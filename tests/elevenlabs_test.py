import os
from RealtimeTTS import TextToAudioStream, ElevenlabsEngine


def dummy_generator():
    yield "This is a sentence. And here's another! Yet, "
    yield "there's more. This ends now."


engine = ElevenlabsEngine(os.environ.get("ELEVENLABS_API_KEY"))
TextToAudioStream(engine).feed(dummy_generator()).play()
