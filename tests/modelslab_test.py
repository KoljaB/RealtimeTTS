"""
Test file for ModelsLab TTS engine.

Usage:
    export MODELSLAB_API_KEY="your_api_key"
    python modelslab_test.py

Requirements:
    pip install requests pyaudio RealtimeTTS
"""
import os
import sys

if __name__ == "__main__":
    from RealtimeTTS import TextToAudioStream, ModelsLabEngine

    # Get API key from environment or prompt
    api_key = os.environ.get("MODELSLAB_API_KEY")
    if not api_key:
        print("Error: MODELSLAB_API_KEY environment variable not set")
        print("Set it with: export MODELSLAB_API_KEY='your_api_key'")
        sys.exit(1)

    def dummy_generator():
        yield "Hello! This is a test of the ModelsLab text to speech engine."
        yield "It supports multiple languages including English, Spanish, French, and more."
        yield "The voices can be customized with speed and emotion parameters."

    # Initialize engine with debug output
    engine = ModelsLabEngine(
        api_key=api_key,
        voice="madison",
        language="american english",
        speed=1.0,
        debug=True
    )

    # List available voices
    print("\nAvailable voices:")
    voices = engine.get_voices()
    for voice in voices[:10]:  # Show first 10
        print(f"  - {voice.name} ({voice.language})")
    print(f"  ... and {len(voices) - 10} more")

    # Create stream
    stream = TextToAudioStream(engine)

    print("\nStarting to play stream...")
    try:
        stream.feed(dummy_generator()).play(log_synthesized_text=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        engine.shutdown()
        print("Done")
