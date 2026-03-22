import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    logging.debug("importing from realtimetts")
    from RealtimeTTS import TextToAudioStream, GTTSEngine, GTTSVoice

    def dummy_generator():
        yield "Hey guys! These here are realtime spoken sentences based on gtts text synthesis. "
        yield "With a voice based on google translate. "

    logging.debug("creating voice")
    voice = GTTSVoice(speed=1.3)
    logging.debug("creating engine")
    engine = GTTSEngine(voice)
    logging.debug("creating stream")
    stream = TextToAudioStream(engine)

    print("Starting to play stream")
    stream.feed(dummy_generator()).play(log_synthesized_text=True)
