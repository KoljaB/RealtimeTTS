if __name__ == "__main__":
    from RealtimeTTS import TextToAudioStream, StyleTTSEngine

    def dummy_generator():
        yield "Hey guys! These here are realtime spoken sentences based on local text synthesis. "
        yield "With a local, neuronal, cloned voice. So every spoken sentence sounds unique."

    # for normal use with minimal logging:
    # engine = StyleTTSEngine("../../Models/LJSpeech/config.yml", "../../Models/LJSpeech/epoch_2nd_00100.pth")
    engine = StyleTTSEngine(
        style_root="../../../",
        model_config_path="Models/LJSpeech/config.yml",
        model_checkpoint_path="Models/LJSpeech/epoch_2nd_00100.pth")

    # test with extended logging:
    # import logging
    # logging.basicConfig(level=logging.INFO)
    # engine = CoquiEngine(level=logging.INFO)

    stream = TextToAudioStream(engine, muted=True)

    print("Starting to play stream")
    stream.feed(dummy_generator())
    stream.play(output_wavfile=stream.engine.engine_name + "_output.wav", muted=True)

    engine.shutdown()
