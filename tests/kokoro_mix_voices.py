from RealtimeTTS import TextToAudioStream, KokoroEngine

# Initialize the engine with af_sky voice
engine = KokoroEngine(default_voice="af_sky")
engine.set_voice("0.714*af_nicole + 0.286*af_sky")

voices = {
    "long_mix": "0.078 * af_bella + 0.431 * af_nicole + 0.284 * af_sky + 0.142 * bf_emma + 0.065 * bf_isabella",
    "mixed": "0.714*af_nicole + 0.286*af_sky",
    "nicole": "af_nicole",
    "sky": "af_sky"
}
    

# Define the longer text to be spoken
long_text = (
    "This is a test of a longer text being spoken with the REPLACE voice. "
    "The engine should smoothly process and play the audio without issues. "
    "RealtimeTTS is designed to handle continuous speech synthesis efficiently, "
    "ensuring natural prosody and pronunciation. Let's see how well it performs."
)

# Set a comfortable speaking speed
engine.set_speed(1.0)  # Adjust speed if needed

for voice_name, voice in voices.items():
    print(f"Testing {voice_name} voice: {voice}")
    engine.set_voice(voice)
    text_to_speak = long_text.replace("REPLACE", voice_name)
    TextToAudioStream(engine).feed(text_to_speak).play(log_synthesized_text=True, output_wavfile=f"af_{voice_name}.wav")

engine.shutdown()