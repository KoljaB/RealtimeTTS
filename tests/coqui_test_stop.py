import time
from RealtimeTTS import TextToAudioStream, CoquiEngine

ttft_values = []
stop_times = []
current_start_time = None

def on_audio_stream_start_callback():
    """Called when the audio stream starts. Computes and prints the TTFT."""
    global current_start_time, ttft_values
    if current_start_time is None:
        print("No current start time set.")
        return
    ttft = time.time() - current_start_time
    ttft_values.append(ttft)
    print(f"<TTFT>: {ttft:.2f} s")

if __name__ == "__main__":
    import logging
    # logging.basicConfig(level=logging.DEBUG)
    engine = CoquiEngine(
        use_deepspeed=True,
        voice="reference_audio.wav",
        thread_count=24,
        stream_chunk_size=8,
        overlap_wav_len=1024,
        # level=logging.DEBUG,
    )
    stream = TextToAudioStream(engine, on_audio_stream_start=on_audio_stream_start_callback)

    # Warm-up: feed a muted short text to prepare the engine.
    current_start_time = time.time()
    stream.feed("warm up").play()

    # Loop through 30 streams.
    for i in range(30):
        print(f"\nStarting Stream {i+1}")
        # Prepare two-sentence text.
        text = f"This is stream {i+1}. Here we test the TTS quick start process."
        current_start_time = time.time()  # Update start time for TTFT calculation.
        
        # Feed the text and start the stream asynchronously.
        stream.feed(text).play_async()
        
        # Let the stream play for 1 second.
        time.sleep(1)
        
        # Measure and record the time needed to stop the stream.
        stop_start_time = time.time()
        stream.stop()
        stop_duration = time.time() - stop_start_time
        stop_times.append(stop_duration)
        print(f"<STOP-TIME>: {stop_duration:.2f} s")

        time.sleep(0.1)  # Small delay to ensure the stream has time to stop before the next iteration.

    # Cleanly shut down the engine.
    engine.shutdown()

    ttft_values.pop(0)  # Remove the first TTFT value (warm-up).

    # Print out all TTFT measurements.
    print("\nTTFT values measured for each stream:")
    for idx, ttft in enumerate(ttft_values, start=1):
        print(f"Stream {idx}: {ttft:.2f} seconds")

    # Print out all stop time measurements.
    print("\nStop time values measured for each stream:")
    for idx, stop_time in enumerate(stop_times, start=1):
        print(f"Stream {idx}: {stop_time:.2f} seconds")
    
    print("\nAll measurements complete.")
    print("Average TTFT:", sum(ttft_values) / len(ttft_values))
    print("Average Stop Time:", sum(stop_times) / len(stop_times))
