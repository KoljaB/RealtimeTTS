"""
1. Create and activate venv:
    python -m venv venv
    venv\Scripts\activate.bat
2. Install dependencies:
    pip install realtimetts[coqui]
3. Update CUDA and install deepspeed for faster processing:
    pip install torch==2.1.2+cu121 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
    pip install https://github.com/daswer123/deepspeed-windows-wheels/releases/download/11.2/deepspeed-0.11.2+cuda121-cp310-cp310-win_amd64.whl
"""

if __name__ == "__main__":
    import time
    from RealtimeTTS import TextToAudioStream, CoquiEngine

    def dummy_generator():
        yield "Hey guys. These here are realtime spoken sentences based on local text synthesis. "
        yield "With a local, neuronal, cloned voice. So every spoken sentence sounds unique."

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

    # for normal use with minimal logging:
    engine = CoquiEngine(use_deepspeed=True)

    # test with extended logging:
    # import logging
    # logging.basicConfig(level=logging.INFO)
    # engine = CoquiEngine(level=logging.INFO)

    start_time = 0
    def on_audio_stream_start_callback():
        global start_time
        delta = time.time() - start_time
        print("<TTFT>", f"Time: {delta:.2f}s")

    stream = TextToAudioStream(engine, on_audio_stream_start=on_audio_stream_start_callback)
    stream.feed("warm up").play(muted=True)

    print("Starting to play stream")
    before_sentence_callback, on_sentence_callback = create_synthesis_callbacks(start_time)
    start_time = time.time()
    stream.feed(dummy_generator()).play(
        log_synthesized_text=True,
        output_wavfile="output.wav",
        before_sentence_synthesized=before_sentence_callback,
        on_sentence_synthesized=on_sentence_callback,
    )
    end_time = time.time()

    print("Playout finished")
    engine.shutdown()
