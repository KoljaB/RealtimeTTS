import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.debug("importing from realtimetts")
    from RealtimeTTS import TextToAudioStream, NeuTTSEngine, NeuTTSVoice

    def dummy_generator():
        yield "Hey guys! These here are realtime spoken sentences based on neu text synthesis. "
        yield "With a cute voice. "

    logging.debug("creating engine")
    engine = NeuTTSEngine(
        device="cuda",
        voices_dir=r"D:\Projekte\TTS\RealtimeTTS\RealtimeTTS_Work\RealtimeTTS\RealtimeTTS\engines\voices" # <-- put the path to your voices directory here
    )

    logging.debug("creating stream")
    stream = TextToAudioStream(engine)

    print("Starting to play stream")
    stream.feed(dummy_generator()).play(log_synthesized_text=True)
