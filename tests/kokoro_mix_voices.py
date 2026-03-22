from RealtimeTTS import TextToAudioStream, KokoroEngine

# Initialize the engine
engine = KokoroEngine(voice="af_sky", debug=True)

# Define the voices we want to test
voices = {
    "long_mix": "0.078 * af_bella + 0.431 * af_nicole + 0.284 * af_sky + 0.142 * bf_emma + 0.065 * bf_isabella",    
    "mixed": "0.714*af_nicole + 0.286*af_sky",
    "nicole": "af_nicole",
    "sky": "af_sky"
}

# Base text with a placeholder for the voice name
base_text = (
    "This is a test of a longer text being spoken with the {voice_name} voice. "
    "The engine should smoothly process and play the audio without issues. "
    "RealtimeTTS is designed to handle continuous speech synthesis efficiently, "
    "ensuring natural prosody and pronunciation. Let's see how well it performs."
)

# Set a comfortable speaking speed
engine.set_speed(1.3)

# Iterate over each voice
for voice_name, voice_expr in voices.items():
    engine.set_voice(voice_expr)
    text_to_speak = base_text.format(voice_name=voice_name)

    print(f"Testing {voice_name} voice: {voice_expr}")
    TextToAudioStream(engine).feed(text_to_speak).play(
        log_synthesized_text=True,
        output_wavfile=f"af_{voice_name}.wav"
    )

# Shutdown the engine when done
engine.shutdown()
