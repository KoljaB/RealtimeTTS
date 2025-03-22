"""
    INSTALLATION:
        pip install realtimetts[orpheus]


    HOW TO RUN:
        Two ways to run the modeL:

        1. On LMStudio:
        - Start LMStudio
        - Load orpheus-3b-0.1-ft-Q8_0-GGUF in Q8_0 quantization
        - Make sure server is running
        
        2. Any other LLM provider:
        - make sure the the server supports completions api
        - load orpheus-3b-0.1-ft-Q8_0-GGUF model in Q8_0 quantization and run the server
        - provide api_url to the server to OrpheusEngine constructor    

    WHAT CAN IT DO:

        You can add the following emotive tags:
            <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>

        Voices available are:
            "tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"
"""


from RealtimeTTS import TextToAudioStream, OrpheusEngine, OrpheusVoice

# Structured data containing voices and their corresponding texts
TEST_CASES = [
    {
        "voice": "zoe",  # Female
        "text": "Don't you just hate it when <laugh> your cat wakes you up like this? Meow. <laugh> Meow. Meow. <chuckle> Meow."
    },
    {
        "voice": "tara",  # Female
        "text": "Asked my assistant to stop talking. Now it's just <laugh> whispering: \"null, null, null...\""
    },
    {
        "voice": "mia",  # Female
        "text": "Told my assistant I need dating pickup lines. It said <laugh> \"Are you a router? Because I'm connecting.\""
    },
    {
        "voice": "jess",  # Male
        "text": "I told my assistant a horror story. It <laugh> got so scared it <chuckle> switched to Comic Sans."
    }
]

def create_generator(text):
    """Create a text generator for a single text string"""
    def generator():
        yield text
    return generator()

def main():
    print("Initializing TTS system...")
    engine = OrpheusEngine()
    stream = TextToAudioStream(engine)

    # Warmup the engine with a short phrase
    print("Performing system warmup...")
    stream.feed(create_generator("System initialization complete"))
    stream.play(muted=True)

    # Process all test cases automatically
    for case in TEST_CASES:
        print(f"\nProcessing voice: {case['voice'].upper()}")
        voice = OrpheusVoice(case["voice"])
        engine.set_voice(voice)
        
        print("Generating audio...")
        stream.feed(create_generator(case["text"]))
        stream.play(log_synthesized_text=True)

    print("\nAll generations completed!")

if __name__ == "__main__":
    main()
