"""Manual 60db smoke: SIXTYDB_API_KEY and SIXTYDB_VOICE_ID are required."""

if __name__ == "__main__":
    import os
    from RealtimeTTS import SixtyDBEngine, TextToAudioStream

    engine = SixtyDBEngine(voice=os.environ["SIXTYDB_VOICE_ID"])
    stream = TextToAudioStream(engine)
    try:
        stream.feed("Hello! This is a test of the 60db text to speech engine.").play()
    finally:
        stream.stop()
        engine.shutdown()
