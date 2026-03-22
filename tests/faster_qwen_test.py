#!/usr/bin/env python
"""
FasterQwen3-TTS test for RealtimeTTS.

Requirements:
    pip install faster-qwen3-tts torch

Note: 
    FasterQwenTTS uses voice cloning (zero-shot) or instruction-based generation.
    You will need a short reference audio file (e.g., 'ref_audio.wav') and its 
    exact text transcript to run this test properly. 
"""

import logging
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

from RealtimeTTS import TextToAudioStream
from RealtimeTTS import FasterQwenEngine, FasterQwenVoice

# --- CONFIGURATION ---
REFERENCE_AUDIO = "ref_audio.wav" 
REFERENCE_TEXT = "In 2013 when the Earth's rotation came to a halt the world called on one man who could make a difference. When it happened again the world called on him once more. Now the one man who made a difference five times before is about to make a difference again. Only this time it's different. Doug Speedman Scorcher six Global Meltdown."
TARGET_LANGUAGE = "English"
TARGET_EMOTION = "Speak with a very excited and enthusiastic voice."


def dummy_generator():
    yield "Hey guys! "
    yield "This is a real-time streaming test using the Faster Qwen 3 engine. "
    yield "Because it utilizes CUDA graphs and static caching, "
    yield "the latency should be incredibly low. "
    yield "It is practically perfect for responsive conversational AI."


if __name__ == "__main__":
    # Check if the reference audio exists to prevent cryptic model errors
    if not os.path.exists(REFERENCE_AUDIO):
        logging.error(f"Reference audio file '{REFERENCE_AUDIO}' not found!")
        print(f"\n⚠️  Please place a short WAV file named '{REFERENCE_AUDIO}' in this directory.")
        print(f"Also ensure you update the 'REFERENCE_TEXT' variable in this script to match what is spoken in the audio.")
        exit(1)

    print("Initializing FasterQwenVoice configuration...")
    # Create the voice object
    my_voice = FasterQwenVoice(
        name="ExcitedClone",
        ref_audio=REFERENCE_AUDIO,
        ref_text=REFERENCE_TEXT,
        language=TARGET_LANGUAGE,
        instruct=TARGET_EMOTION
    )

    print("Loading FasterQwenEngine (this might take a few seconds on the first run)...")
    # Initialize the engine (0.6B model is great for tests due to speed and lower VRAM)
    engine = FasterQwenEngine(
        model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device="cuda",
        voice=my_voice,
        chunk_size=8,        # 8 steps ≈ 667ms audio chunks. Lower = faster TTFA but more overhead
        xvec_only=True,      # True = cleaner language switching & lower latency (Speaker embedding only)
        # debug=True           # Enable debug to see the TTFA metrics print out
    )

    print(f"Using voice clone reference: {REFERENCE_AUDIO}")
    print("Starting playback stream...")
    
    start_time = time.time()

    def on_audio_stream_start():
        delta = time.time() - start_time
        print(f"\n<TTFA - Time To First Audio> {delta:.3f}s")

    # Initialize the RealtimeTTS stream
    stream = TextToAudioStream(
        engine,
        on_audio_stream_start=on_audio_stream_start,
    )

    # Feed the generator and play the audio
    stream.feed(dummy_generator()).play(log_synthesized_text=True)

    # Cleanup
    engine.shutdown()
    print("\nTest completed successfully.")