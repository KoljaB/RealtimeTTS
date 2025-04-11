if __name__ == "__main__":
    import time 
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

    def create_synthesis_callbacks(start_time):
        # Use a local variable to store the synthesis start time
        sentence_synth_start = None

        def before_sentence_callback(_):
            nonlocal sentence_synth_start
            sentence_synth_start = time.time()
            elapsed = sentence_synth_start - start_time
            print("<SYNTHESIS_START>", f"{elapsed:.2f}s")

        def on_sentence_callback(_):
            if sentence_synth_start is not None:
                delta = time.time() - sentence_synth_start
                print("<SYNTHESIS_DONE>", f"Delta: {delta:.2f}s")
            else:
                print("<SYNTHESIS_DONE>", "No start time recorded.")
        return before_sentence_callback, on_sentence_callback

    # Adjust these paths to your local setup
    styletts_root = "D:/Dev/StyleTTS_Realtime/StyleTTS2"

    # Create StyleTTSVoice instances for both models
    voice_1 = StyleTTSVoice(
        model_config_path="D:/Data/Models/style/Nicole/config.yml",
        model_checkpoint_path="D:/Data/Models/style/Nicole/epoch_2nd_00036.pth",
        ref_audio_path="D:/Data/Models/style/Nicole/file___1_file___1_segment_98.wav"
    )

    voice_2 = StyleTTSVoice(
        model_config_path="D:/Data/Models/style/LongLasi/Lasinya_config.yml",
        model_checkpoint_path="D:/Data/Models/style/LongLasi/Lasinya_00085.pth",
        ref_audio_path="D:/Data/Models/style/LongLasi/reference.wav"
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
    start_time = 0
    def on_audio_stream_start_callback():
        global start_time
        delta = time.time() - start_time
        print("<TTFT>", f"Time: {delta:.2f}s")
    stream = TextToAudioStream(engine, on_audio_stream_start=on_audio_stream_start_callback)

    # Play with the first model
    print("Playing with the first model...")
    stream.feed(dummy_generator_1())
    start_time = time.time()
    before_sentence_callback, on_sentence_callback = create_synthesis_callbacks(start_time)
    stream.play(log_synthesized_text=True, before_sentence_synthesized=before_sentence_callback, on_sentence_synthesized=on_sentence_callback)

    # Switch to the second voice at runtime
    print("\nSwitching to the second model...")
    engine.set_voice(voice_2)  # Use set_voice to update the voice configuration

    # Play with the second model
    print("Playing with the second model...")
    stream.feed(dummy_generator_2())
    engine.diffusion_steps = 3
    start_time = time.time()
    before_sentence_callback, on_sentence_callback = create_synthesis_callbacks(start_time)
    stream.play(log_synthesized_text=True, before_sentence_synthesized=before_sentence_callback, on_sentence_synthesized=on_sentence_callback)

    # Switch to the third voice
    print("\nSwitching to the third model...")
    engine.set_voice(voice_3)  # Switch to the third voice configuration

    # Play again with the first model
    print("Playing with the first model again...")
    stream.feed(dummy_generator_3())
    engine.diffusion_steps = 50
    start_time = time.time()
    before_sentence_callback, on_sentence_callback = create_synthesis_callbacks(start_time)
    stream.play(log_synthesized_text=True, before_sentence_synthesized=before_sentence_callback, on_sentence_synthesized=on_sentence_callback)

    # Shutdown the engine
    engine.shutdown()
