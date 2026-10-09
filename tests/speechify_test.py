"""
Speechify TTS test for RealtimeTTS.

Requirements:
    pip install realtimetts[speechify]

Set your API key:
    export SPEECHIFY_API_KEY=your_key_here  (Linux/Mac)
    set SPEECHIFY_API_KEY=your_key_here     (Windows)
"""

import time
from RealtimeTTS import TextToAudioStream, SpeechifyEngine


def dummy_generator():
    yield "Hey guys! These here are realtime spoken sentences using Speechify streaming. "
    yield "This should arrive in chunks and be played immediately."


if __name__ == "__main__":
    engine = SpeechifyEngine(voice_id="geffen_32", model="simba-3.2", debug=True)

    start_time = time.time()

    def on_audio_stream_start():
        print(f"<TTFT> {time.time() - start_time:.2f}s")

    stream = TextToAudioStream(engine, on_audio_stream_start=on_audio_stream_start)
    stream.feed(dummy_generator()).play(log_synthesized_text=True)

    engine.shutdown()
