use_logging = False

xtts_model = "models/xtts/Lasinya"
xtts_voice = "Lasinya_Reference.json"
rvc_model = "models/rvc/Lasinya"


if __name__ == "__main__":
    from xtts_rvc_synthesizer import XTTSRVCSynthesizer
    import time

    print("Starting synthesizer")

    tts = XTTSRVCSynthesizer(
        xtts_model=xtts_model,
        xtts_voice=xtts_voice,
        rvc_model=rvc_model,
        rvc_sample_rate=40000,
        use_logging=use_logging,
    )

    print("Pushing text")
    tts.push_text(
        "Hello World! I like to see this. I hope this get's synthesized well. "
    )
    time.sleep(5)

    print("Stop")
    tts.stop()

    print("Pushing text char by char")
    text = "This is another sentence. We will add it to the synthesis queue. "
    for char in text:
        time.sleep(0.1)
        tts.push_text(char)

    time.sleep(3)

    print("Synthesizing")
    tts.synthesize()

    print("Add another text char by char")
    text = "Now we will add this. If this succeeds I am very happy. "
    for char in text:
        time.sleep(0.1)
        tts.push_text(char)

    print("Wait 5 sec")
    time.sleep(5)

    print("Synthesizing")
    tts.synthesize()

    print("Wait for finish playout")
    tts.wait_playing()

    print("Finished")
    tts.shutdown()

    print("Exit")
