if __name__ == "__main__":
    import os
    from RealtimeTTS import TextToAudioStream, AzureEngine

    def dummy_generator():
        yield "Hey guys! These here are realtime spoken sentences based on local text synthesis. "
        yield "With a local, neuronal, cloned voice. So every spoken sentence sounds unique."

    # for normal use with minimal logging:
    import os
    engine = AzureEngine(
        os.environ["AZURE_SPEECH_KEY"],
        os.environ["AZURE_SPEECH_REGION"],
        audio_format="riff-48khz-16bit-mono-pcm"
    )

    stream = TextToAudioStream(engine)

    print("Starting to play stream")
    stream.feed(dummy_generator()).play(log_synthesized_text=True)

    engine.shutdown()
