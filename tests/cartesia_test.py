#!/usr/bin/env python
"""
Cartesia TTS test for RealtimeTTS.

Requirements:
    pip install cartesia[websockets]

Set your API key:
    export CARTESIA_API_KEY=your_key_here  (Linux/Mac)
    set CARTESIA_API_KEY=your_key_here     (Windows)
"""

# from cartesia import Cartesia
# import os

# client = Cartesia(api_key=os.environ["CARTESIA_API_KEY"])

# for voice in client.voices.list():
#     print(voice.name, voice.id)

# exit(0)

import logging
logging.basicConfig(level=logging.INFO)

import os
import time
from RealtimeTTS import TextToAudioStream, CartesiaEngine


def dummy_generator():
    yield "Hey guys! These here are realtime spoken sentences using Cartesia streaming. "
    yield "This should arrive in chunks and be played immediately. "
    yield "If everything works, latency should be quite low."


if __name__ == "__main__":
    api_key = os.environ.get("CARTESIA_API_KEY", "")

    if not api_key:
        raise ValueError("Please set CARTESIA_API_KEY environment variable")

    engine = CartesiaEngine(
        api_key=api_key,
        model_id="sonic-3",
        voice_id="e07c00bc-4134-4eae-9ea4-1a55fb45746b",  # <-- put a valid voice id here
        output_format={
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": 44100,
        },
        debug=True,
    )

    print("Fetching available voices...")
    voices = engine.get_voices()

    if not voices:
        raise RuntimeError("No voices returned from Cartesia")

    # Pick first voice if none set
    if not engine.voice_id:
        engine.set_voice(voices[0])
        print(f"Using voice: {voices[0]}")

    stream = TextToAudioStream(engine)

    print("Starting playback...")
    start_time = time.time()

    def on_audio_stream_start():
        delta = time.time() - start_time
        print(f"<TTFT> {delta:.2f}s")

    stream = TextToAudioStream(
        engine,
        on_audio_stream_start=on_audio_stream_start,
    )

    stream.feed(dummy_generator()).play(log_synthesized_text=True)

    engine.shutdown()
