"""
    pip install realtimetts[orpheus]

    Two ways to run the modeL:

    1. On LMStudio:
      - Start LMStudio
      - Load orpheus-3b-0.1-ft-Q8_0-GGUF in Q8_0 quantization
      - Make sure server is running
    
    2. Any other LLM provider:
      - make sure the the server supports completions api
      - load orpheus-3b-0.1-ft-Q8_0-GGUF model in Q8_0 quantization and run the server
      - provide api_url to the server to OrpheusEngine constructor

"""
if __name__ == "__main__":
    print("Starting orpheus test...")
    from RealtimeTTS import TextToAudioStream, OrpheusEngine, OrpheusVoice

    # You can additionally add the following emotive tags:
    # <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>

    def dummy_generator():
        yield "This is <gasp> orpheus t t s speaking."

    def dummy_generator2():
        yield "I just hope that <laugh> you're all in as good a mood as I am!"

    print("Creating engine...")
    engine = OrpheusEngine()
    stream = TextToAudioStream(engine)

    print("Setting voice...")
    voice = OrpheusVoice("zac")
    engine.set_voice(voice)

    # warmup
    print("Warming up engine...")
    stream.feed("warming up engine")
    stream.play(muted=True)

    # wait for key press
    input("Ready - press enter to start tts generation...")
    print("Playing tts generation...")
    stream.feed(dummy_generator())
    stream.play()

    print("Setting voice...")
    voice = OrpheusVoice("zoe")
    engine.set_voice(voice)

    # wait for key press
    input("Ready - press enter for another generation...")
    print("Playing another generation...")
    stream.feed(dummy_generator2())
    stream.play()
