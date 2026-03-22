import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.debug("importing from realtimetts")
    from RealtimeTTS import TextToAudioStream, NeuTTSEngine, NeuTTSVoice

    def dummy_generator():
        yield "Hey guys! These here are realtime spoken sentences based on neu text synthesis. "
        yield "With a cute voice. "

        

    # logging.debug("creating voice")
    # voice = NeuTTSVoice()

    logging.debug("creating engine")
    engine = NeuTTSEngine(
        device="cuda",
        voices_dir=r"D:\Projekte\TTS\RealtimeTTS\RealtimeTTS_Work\RealtimeTTS\RealtimeTTS\engines\voices"
    )

    # engine = NeuTTSEngine(
    # )
    # voice = engine.add_voice(
    #     name="jo",
    #     audio_path="D:/path/to/neutts/samples/jo.wav",
    #     transcript=open("D:/path/to/neutts/samples/jo.txt").read().strip()
    # )

    # engine.set_voice(voice)

    logging.debug("creating stream")
    stream = TextToAudioStream(engine)

    print("Starting to play stream")
    stream.feed(dummy_generator()).play(log_synthesized_text=True)
