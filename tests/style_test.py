if __name__ == "__main__":
    from RealtimeTTS import TextToAudioStream, StyleTTSEngine, StyleTTSVoice

    def dummy_generator_1():
        yield "This is the first voice model speaking. "
        yield "The elegance of the style and its flow is simply captivating. "
        yield "We’ll soon switch to another model. "

    def dummy_generator_2():
        yield "And here we are! "
        yield "You’re now listening to the second voice model, with a different style and tone. "
        yield "It’s fascinating how StyleTTS can adapt seamlessly. "

    def dummy_generator_3():
        yield "Welcome back again! "
        yield "We’re testing the third voice model now. "
        yield "The transition between styles is smooth and effortless. "

    # Adjust these paths to your local setup
    styletts_root = "D:/Dev/StyleTTS_Realtime/StyleTTS2"

    # Create StyleTTSVoice instances for both models
    voice_1 = StyleTTSVoice(
        model_config_path="D:/Data/Models/style/Nicole/config.yml",
        model_checkpoint_path="D:/Data/Models/style/Nicole/epoch_2nd_00036.pth",
        ref_audio_path="D:/Data/Models/style/Nicole/file___1_file___1_segment_98.wav"
    )

    voice_2 = StyleTTSVoice(
        model_config_path="D:/Data/Models/style/LongLasi/LongLasi_config.yml",
        model_checkpoint_path="D:/Data/Models/style/LongLasi/epoch_2nd_00047.pth",
        ref_audio_path="D:/Data/Models/style/LongLasi/file___1_file___1_segment_116.wav"
    )

    voice_3 = StyleTTSVoice(
        model_config_path="D:/Data/Models/style/ExtLasi/ExcLasi_config.yml",
        model_checkpoint_path="D:/Data/Models/style/ExtLasi/epoch_2nd_00039.pth",
        ref_audio_path="D:/Data/Models/style/ExtLasi/file___1_file___1_segment_33.wav"
    )

    # Initialize the engine with the first voice
    engine = StyleTTSEngine(
        style_root=styletts_root,
        voice=voice_1,  # Pass the first StyleTTSVoice instance
        alpha=0.3,
        beta=1.0,
        diffusion_steps=10,
        embedding_scale=1.0,
        cuda_reset_delay=0.0,  # Custom delay for CUDA reset
    )

    # Create a TextToAudioStream with the engine
    stream = TextToAudioStream(engine)

    # Play with the first model
    print("Playing with the first model...")
    stream.feed(dummy_generator_1())
    stream.play(log_synthesized_text=True)

    # Switch to the second voice at runtime
    print("\nSwitching to the second model...")
    engine.set_voice(voice_2)  # Use set_voice to update the voice configuration

    # Play with the second model
    print("Playing with the second model...")
    stream.feed(dummy_generator_2())
    stream.play(log_synthesized_text=True)

    # Switch to the third voice
    print("\nSwitching to the third model...")
    engine.set_voice(voice_3)  # Switch to the third voice configuration

    # Play again with the first model
    print("Playing with the first model again...")
    stream.feed(dummy_generator_3())
    stream.play(log_synthesized_text=True)

    # Shutdown the engine
    engine.shutdown()
