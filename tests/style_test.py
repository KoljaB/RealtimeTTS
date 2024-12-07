if __name__ == "__main__":
    from RealtimeTTS import TextToAudioStream, StyleTTSEngine

    def dummy_generator():
        yield "Close your eyes for a moment... can you hear it? "
        yield "That’s not just a voice — it’s StyleTTS2, turning words into experiences with depth, charm, and a hint of allure. "
        yield "Every word feels intentional, like it was crafted just for you. "
        yield "Here’s the exciting part: it’s joining the RealtimeTTS library soon. "
        yield "Prepare yourself for real-time expressive speech that’s as authentic as the moment you’re in. "

    # adjust these paths to your local setup (stylett2 installation folder, model config, model checkpoint, reference audio)
    styletts_root = "D:/Dev/StyleTTS_Realtime/StyleTTS2"
    model_config_path = "D:/Dev/StyleTTS_Realtime/StyleTTS2/Models/Nicole/config.yml"
    model_checkpoint_path = "D:/Dev/StyleTTS_Realtime/StyleTTS2/Models/Nicole/epoch_2nd_00036.pth"
    ref_audio_path = "D:/Dev/StyleTTS_Realtime/RealtimeTTS/tests/nicole.wav"
    
    engine = StyleTTSEngine(
        style_root=styletts_root,
        model_config_path=model_config_path,
        model_checkpoint_path=model_checkpoint_path,
        ref_audio_path=ref_audio_path,
        alpha=0.3,
        beta=0.7,
        diffusion_steps=50,
        embedding_scale=1,)

    stream = TextToAudioStream(engine)

    print("Starting to play stream")
    stream.feed(dummy_generator())
    stream.play(log_synthesized_text=True)

    engine.shutdown()
