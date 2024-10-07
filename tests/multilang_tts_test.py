if __name__ == '__main__':
    import os
    from RealtimeTTS import TextToAudioStream

    def dummy_generator(language):
        if language == "zh":
            yield "我喜欢读书。天气很好。我们去公园吧。今天是星期五。早上好。这是我的朋友。请帮我。吃饭了吗？我在学中文。晚安。"
        elif language == "en":
            yield "I love reading. The weather is nice. Let's go to the park. Today is Friday. Good morning. This is my friend. Please help me. Have you eaten? I'm learning English. Good night."
        elif language == "es":
            yield "Me gusta leer. Hace buen tiempo. Vamos al parque. Hoy es viernes. Buenos días. Este es mi amigo. Por favor, ayúdame. ¿Has comido? Estoy aprendiendo español. Buenas noches."
        elif language == "de":
            yield "Ich liebe es zu lesen. Das Wetter ist schön. Lass uns in den Park gehen. Heute ist Freitag. Guten Morgen. Das ist mein Freund. Bitte hilf mir. Hast du gegessen? Ich lerne Deutsch. Gute Nacht."

    def synthesize(engine, language, generator):
        stream = TextToAudioStream(engine)

        print(f"Starting to play stream in {language}")
        stream.feed(generator(language))
        filename = f"synthesis_{language}_" + engine.engine_name

        # Adjust the tokenizer and language for each language
        tokenizer = "stanza" if language in ["zh", "es", "de"] else None
        stream.play(
            minimum_sentence_length=2,
            minimum_first_fragment_length=2, 
            output_wavfile=f"{filename}.wav",
            on_sentence_synthesized=lambda sentence: 
                print(f"Synthesized ({language}): " + sentence),
            tokenizer=tokenizer, 
            language=language,
            context_size=2
        )

        with open(f"{filename}.txt", "w", encoding="utf-8") as f:
            f.write(stream.text())

        engine.shutdown()

    def get_engine(name, language):
        if name == "coqui":
            from RealtimeTTS import CoquiEngine
            voices = {
                "zh": "female_chinese",
                "en": "female_english",
                "es": "female_spanish",
                "de": "female_german"
            }
            return CoquiEngine(voice=voices[language], language=language)
        elif name == "azure":
            from RealtimeTTS import AzureEngine
            voices = {
                "zh": "zh-CN-XiaoxiaoNeural",
                "en": "en-US-JennyNeural",
                "es": "es-ES-ElviraNeural",
                "de": "de-DE-KatjaNeural"
            }
            return AzureEngine(os.environ.get("AZURE_SPEECH_KEY"), os.environ.get("AZURE_SPEECH_REGION"), voice=voices[language])
        elif name == "elevenlabs":
            from RealtimeTTS import ElevenlabsEngine
            return ElevenlabsEngine(os.environ.get("ELEVENLABS_API_KEY"))
        else:
            from RealtimeTTS import SystemEngine
            voices = {
                "zh": "Huihui",
                "en": "Microsoft Zira",
                "es": "Microsoft Helena",
                "de": "Microsoft Hedda"
            }
            return SystemEngine(voice=voices[language])

    languages = ["zh", "en", "es", "de"]
    
    for engine_name in ["coqui", "elevenlabs", "azure", "system"]:
        for language in languages:
            print(f"Starting engine: {engine_name} for language: {language}")
            engine = get_engine(engine_name, language)
            print(f"Synthesizing with engine: {engine_name} for language: {language}")
            synthesize(engine, language, dummy_generator)
