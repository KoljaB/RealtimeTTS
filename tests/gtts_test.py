if __name__ == "__main__":
    from RealtimeTTS import TextToAudioStream, GTTSEngine, GTTSVoice

    def dummy_generator():
        yield "Hey guys! These here are realtime spoken sentences based on gtts text synthesis. "
        yield "With a voice based on google translate. "

    voice = GTTSVoice(speed=1.3)
    engine = GTTSEngine(voice)
    stream = TextToAudioStream(engine)

    print("Starting to play stream")
    stream.feed(dummy_generator()).play(log_synthesized_text=True)
