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

    import string
    last_word = None
    def process_word(word):
        global last_word
        if last_word and word.word not in set(string.punctuation):
            print(" ", end="", flush=True)

        print(f"{word.word}", end="", flush=True)
        last_word = word.word

    engine.set_voice("Ava")

    stream = TextToAudioStream(engine, on_word=process_word)

    print("Starting to play stream")
    stream.feed(dummy_generator()).play(output_wavfile = "output.wav")

    engine.shutdown()
