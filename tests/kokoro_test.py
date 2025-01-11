if __name__ == "__main__":
    from RealtimeTTS import TextToAudioStream, KokoroEngine

    def dummy_generator():
        yield "This is the first voice model speaking. "
        yield "The elegance of the style and its flow is simply captivating. "
        yield "We’ll soon switch to another voice model. "

    def dummy_generator_2():
        yield "And here we are! "
        yield "You’re now listening to the second voice model, with a different style and tone. "
        yield "It’s fascinating how Kokoro can adapt seamlessly. "

    # Adjust these paths to your local setup
    kokoro_root = "D:/Dev/Kokoro/Kokoro-82M"

    # Initialize the engine with the first voice
    engine = KokoroEngine(
        kokoro_root=kokoro_root,
    )

    # Create a TextToAudioStream with the engine
    stream = TextToAudioStream(engine)

    # Play with the first model
    print("Playing with the first model...")
    stream.feed(dummy_generator())
    stream.play(log_synthesized_text=True)

    engine.set_voice("af_sky")
    # Pick one of: 
    # "af_nicole", 
    # "af",
    # "af_bella",
    # "af_sarah",
    # "am_adam",
    # "am_michael",
    # "bf_emma",
    # "bf_isabella",
    # "bm_george",
    # "bm_lewis",
    # "af_sky"    
    stream.feed(dummy_generator_2())
    stream.play(log_synthesized_text=True)
    
    # Shutdown the engine
    engine.shutdown()
