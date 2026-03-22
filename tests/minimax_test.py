"""
MiniMax Cloud TTS – Quick Test

A simple test that synthesizes text using MiniMax's T2A v2 API.

Env vars: MINIMAX_API_KEY
"""

if __name__ == "__main__":
    import os
    import sys
    import time
    from dotenv import load_dotenv

    load_dotenv()

    from RealtimeTTS import TextToAudioStream, MiniMaxEngine

    # Initialise engine
    engine = MiniMaxEngine(
        model="speech-2.8-hd",
        voice="English_Graceful_Lady",
        debug=True,
    )
    stream = TextToAudioStream(engine)

    print("MiniMax TTS engine ready.")
    print()

    # Simple synthesis test
    test_text = "Hello! This is a test of the MiniMax text to speech engine."
    print(f"Synthesizing: {test_text}")
    stream.feed(test_text)
    stream.play_async()
    while stream.is_playing():
        time.sleep(0.1)

    print()

    # Test voice switching
    print("Switching to English_Persuasive_Man voice...")
    engine.set_voice("English_Persuasive_Man")
    test_text2 = "Now I am speaking with a different voice preset."
    print(f"Synthesizing: {test_text2}")
    stream.feed(test_text2)
    stream.play_async()
    while stream.is_playing():
        time.sleep(0.1)

    print()

    # Interactive mode
    print("Type text to synthesize (or 'quit' to exit):")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() == "quit":
            break

        stream.feed(user_input)
        stream.play_async()
        while stream.is_playing():
            time.sleep(0.1)

    stream.stop()
    engine.shutdown()
    print("Done!")
