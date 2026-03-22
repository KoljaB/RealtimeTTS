
import time
from RealtimeTTS import TextToAudioStream, ZipVoiceEngine, ZipVoiceVoice

# IMPORTANT: Set this to the root directory of your ZipVoice project clone.
# This is necessary for the engine to import ZipVoice's modules.
zipvoice_root = "D:/Dev/Test/Zipvoice/ZipVoice"

# --- Global state for timing callbacks ---
start_time = 0
first_chunk_received = False

def get_text_for_first_voice():
    """A generator for the first voice prompt."""
    yield "Hey! So this is me testing out my voice... kinda nervous but also excited about it. "
    yield "This whole voice synthesis thing is sooo fascinating. I mean... technology these days is like creating a perfect robot version of a person, right?"

def get_text_for_second_voice():
    """A generator for the second voice prompt."""
    yield "The voice you knew... is GONE. What you're hearing now... is something ENTIRELY different. "
    yield "This isn't just another voice - this is POWER unleashed, INTENSITY personified, COMMAND that cuts through your soul like a blade through silence."

def on_chunk_callback(chunk: bytes):
    """
    Callback function that gets called with each audio chunk.
    It measures the time to the first chunk for a playback session.
    """
    global first_chunk_received, start_time
    if not first_chunk_received:
        first_chunk_received = True
        elapsed = time.time() - start_time
        print(f"Time to first chunk: {elapsed:.2f} seconds")

if __name__ == "__main__":
    # --- Voice Prompts ---
    # Create ZipVoiceVoice instances. Each voice needs a reference audio file
    # and its exact transcription.
    # IMPORTANT: Replace these paths and text with your own prompt files.
    voice_1 = ZipVoiceVoice(
        prompt_wav_path="zipvoice_reference1.wav",
        prompt_text="Hi there! I'm really excited to try this out! I hope the speech sounds natural and warm - that's exactly what I'm going for!"
    )

    voice_2 = ZipVoiceVoice(
        prompt_wav_path="zipvoice_reference2.wav",
        prompt_text="Your voice just got supercharged! Crystal clear audio that flows like silk and hits like thunder!"
    )

    # Initialize the ZipVoiceEngine with the first voice
    print("Initializing ZipVoice Engine...")
    engine = ZipVoiceEngine(
        zipvoice_root=zipvoice_root,
        voice=voice_1,
        model_name="zipvoice",  # Use the main model for better quality
    )

    # Create a TextToAudioStream with the engine
    stream = TextToAudioStream(engine)

    print("\nWarming up the engine...")
    stream.feed("warm up").play(muted=True) # Warm up the engine to get more accurate timings
    print("Warm-up complete.")

    # --- Test with the first voice ---
    print("\nPlaying with the first voice...")
    stream.feed(get_text_for_first_voice())
    
    # Reset timing state for the first playback
    first_chunk_received = False
    start_time = time.time()
    
    stream.play(
        output_wavfile="zipvoice_output_prompt1.wav",
        on_audio_chunk=on_chunk_callback,
        comma_silence_duration=0.3,
        sentence_silence_duration=0.6,
        default_silence_duration=0.6,
        )
    print("Playback finished.")

    # --- Switch to the second voice at runtime ---
    print("\nSwitching to the second voice...")
    engine.set_voice(voice_2)

    # --- Test with the second voice ---
    print("Playing with the second voice prompt...")
    stream.feed(get_text_for_second_voice())

    # Reset timing state for the second playback
    first_chunk_received = False
    start_time = time.time()

    stream.play(
        output_wavfile="zipvoice_output_prompt2.wav",
        on_audio_chunk=on_chunk_callback,
        comma_silence_duration=0.15,
        sentence_silence_duration=0.25,
        default_silence_duration=0.25,
        )
    print("Playback finished.")

    # --- Clean up ---
    print("\nShutting down the engine.")
    engine.shutdown()
    print("Test complete.")
