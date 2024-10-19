if __name__ == "__main__":
    import os
    from RealtimeTTS import TextToAudioStream

    def dummy_generator():
        yield "我喜欢读书。天气很好。我们去公园吧。今天是星期五。早上好。这是我的朋友。请帮我。吃饭了吗？我在学中文。晚安。"

    def synthesize(engine, generator):
        stream = TextToAudioStream(engine)

        print("Starting to play stream")
        stream.feed(generator)
        filename = "synthesis_chinese_" + engine.engine_name

        # ❗ use these for chinese: minimum_sentence_length = 2, minimum_first_fragment_length = 2, tokenizer="stanza", language="zh", context_size=2
        stream.play(
            minimum_sentence_length=2,
            minimum_first_fragment_length=2,
            output_wavfile=f"{filename}.wav",
            on_sentence_synthesized=lambda sentence: print("Synthesized: " + sentence),
            tokenizer="stanza",
            language="zh",
            context_size=2,
        )

        with open(f"{filename}.txt", "w", encoding="utf-8") as f:
            f.write(stream.text())

        engine.shutdown()

    def get_engine(name):
        if name == "coqui":
            from RealtimeTTS import CoquiEngine

            # ❗ use these for chinese: voice="female_chinese", language = "zh"   # you can exchange voice with you own
            return CoquiEngine(
                voice="female_chinese", language="zh"
            )  # using a chinese cloning reference gives better quality

        elif name == "azure":
            from RealtimeTTS import AzureEngine

            # ❗ use these for chinese: voice="zh-CN-XiaoxiaoNeural"   # or specify a different azure zn-CN voice
            return AzureEngine(
                os.environ.get("AZURE_SPEECH_KEY"),
                os.environ.get("AZURE_SPEECH_REGION"),
                voice="zh-CN-XiaoxiaoNeural",
            )

        elif name == "elevenlabs":
            from RealtimeTTS import ElevenlabsEngine

            return ElevenlabsEngine(os.environ.get("ELEVENLABS_API_KEY"))

        else:
            from RealtimeTTS import SystemEngine

            # ❗ use these for chinese: voice = "Huihui"   # or specify a different locally installed chinese tts voice
            return SystemEngine(voice="Huihui")

    for engine_name in ["coqui", "elevenlabs", "azure", "system"]:
        print("Starting engine: " + engine_name)
        engine = get_engine(engine_name)

        print("Synthesizing with engine: " + engine_name)
        synthesize(engine, dummy_generator())
