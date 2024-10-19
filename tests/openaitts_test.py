from RealtimeTTS import TextToAudioStream, OpenAIEngine


def dummy_generator():
    yield "Hey guys! "
    yield "These here are "
    yield "realtime spoken words "
    yield "based on openai "
    yield "tts text synthesis."


engine = OpenAIEngine(model="tts-1", voice="nova")
stream = TextToAudioStream(engine)
stream.feed(dummy_generator())

print("Synthesizing...")
stream.play()
