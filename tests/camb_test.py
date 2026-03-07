if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from RealtimeTTS import TextToAudioStream, CambEngine

    engine = CambEngine()

    stream = TextToAudioStream(engine)

    print("Starting CAMB AI TTS stream...")

    stream.feed("Hello, this is a test of the CAMB AI text to speech engine using the MARS model.")
    stream.play_async()

    import time
    while stream.is_playing():
        time.sleep(0.1)

    print("Stream finished.")
