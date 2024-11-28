if __name__ == "__main__":
    from RealtimeTTS import TextToAudioStream, OpenAIEngine

    def dummy_generator():
        yield "Hey guys! These here are realtime spoken sentences based on local text synthesis. "
        yield "With a local, neuronal, cloned voice. So every spoken sentence sounds unique."

    # for normal use with minimal logging:
    engine = OpenAIEngine()

    # test with extended logging:
    # import logging
    # logging.basicConfig(level=logging.INFO)
    # engine = CoquiEngine(level=logging.INFO)

    stream = TextToAudioStream(engine)

    print("Starting to play stream")
    stream.feed(dummy_generator()).play(log_synthesized_text=True)

    engine.shutdown()
