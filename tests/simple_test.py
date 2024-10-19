from RealtimeTTS import TextToAudioStream, SystemEngine


def dummy_generator():
    yield "This is a sentence. And here's another! Yet, "
    yield "there's more. This ends now."


TextToAudioStream(SystemEngine()).feed(dummy_generator()).play()
