if __name__ == "__main__":
    print("Starting Qwen server speed demo...")
    import os
    from RealtimeTTS import QwenServerEngine, TextToAudioStream

    server = os.getenv("QWEN_TTS_SERVER", "http://127.0.0.1:18085")
    print("Connecting to Qwen server...")
    engine = QwenServerEngine(
        server, voice=os.getenv("QWEN_TTS_VOICE", "alex"),
        api_key=os.getenv("REALTIMETTS_API_KEY"), seed=42,
    )
    try:
        stream = TextToAudioStream(
            engine, speed_tags=True,
            on_audio_stream_start=lambda: print("Playing audio..."),
        )
        text = (
            "This is normal speed. "
            '<speed="0.7">This is slower, with the same pitch.</speed> '
            "[sr=1.4]This is faster, with the same pitch.</sr> "
            "And now we are back to normal speed."
        )
        print("Generating speech: normal, scoped 0.7x and 1.4x, then normal")
        stream.feed(text).play(
            before_sentence_synthesized=lambda text: print(f"Synthesizing: {text}"),
        )
        if engine.last_error is not None:
            raise engine.last_error
        if stream.tempo_timeline.output_duration <= 0:
            raise RuntimeError("No audio received from Qwen.")
        print("Done.")
    finally:
        engine.shutdown()
