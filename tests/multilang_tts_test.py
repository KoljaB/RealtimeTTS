if __name__ == "__main__":
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
        elif language == "fr":
            yield "J'aime lire. Le temps est agréable. Allons au parc. Aujourd'hui, c'est vendredi. Bonjour. C'est mon ami. Aidez-moi s'il vous plaît. As-tu mangé? J'apprends le français. Bonne nuit."
        elif language == "it":
            yield "Mi piace leggere. Il tempo è bello. Andiamo al parco. Oggi è venerdì. Buongiorno. Questo è il mio amico. Per favore aiutami. Hai mangiato? Sto imparando l'italiano. Buonanotte."
        elif language == "ja":
            yield "私は読書が好きです。天気がいいですね。公園に行きましょう。今日は金曜日です。おはようございます。これは私の友達です。助けてください。食べましたか？私は日本語を勉強しています。おやすみなさい。"
        elif language == "ko":
            yield "저는 독서를 좋아해요. 날씨가 좋네요. 공원에 가요. 오늘은 금요일이에요. 좋은 아침이에요. 이 분은 제 친구예요. 제발 도와주세요. 식사하셨어요? 저는 영어를 배우고 있어요. 안녕히 주무세요."
        elif language == "ar":
            yield "أحب القراءة. الطقس لطيف. هيا نذهب إلى الحديقة. اليوم هو الجمعة. صباح الخير. هذا صديقي. أرجوك ساعدني. هل تناولت الطعام؟ أنا أتعلم العربية. تصبح على خير."
        elif language == "hi":
            yield "मुझे पढ़ना पसंद है। मौसम अच्छा है। चलो पार्क चलते हैं। आज शुक्रवार है। सुप्रभात। यह मेरा दोस्त है। कृपया मेरी मदद करें। क्या आपने खाना खाया? मैं हिंदी सीख रहा हूं। शुभ रात्रि।"
        elif language == "pt":
            yield "Eu gosto de ler. O tempo está bom. Vamos ao parque. Hoje é sexta-feira. Bom dia. Este é meu amigo. Por favor, me ajude. Você já comeu? Estou aprendendo português. Boa noite."
        elif language == "ru":
            yield "Я люблю читать. Погода хорошая. Пойдем в парк. Сегодня пятница. Доброе утро. Это мой друг. Пожалуйста, помогите мне. Вы уже ели? Я учу русский язык. Спокойной ночи."
        elif language == "id":
            yield "Saya suka membaca. Cuacanya bagus. Ayo pergi ke taman. Hari ini hari Jumat. Selamat pagi. Ini teman saya. Tolong bantu saya. Apakah Anda sudah makan? Saya sedang belajar bahasa Indonesia. Selamat malam."
        elif language == "tr":
            yield "Okumayı seviyorum. Hava güzel. Haydi parka gidelim. Bugün Cuma. Günaydın. Bu benim arkadaşım. Lütfen bana yardım edin. Yemek yediniz mi? Türkçe öğreniyorum. İyi geceler."
        elif language == "vi":
            yield "Tôi thích đọc sách. Thời tiết đẹp. Hãy đi công viên. Hôm nay là thứ Sáu. Chào buổi sáng. Đây là bạn tôi. Làm ơn giúp tôi. Bạn đã ăn chưa? Tôi đang học tiếng Việt. Chúc ngủ ngon."
        elif language == "bn":
            yield "আমি পড়তে পছন্দ করি। আবহাওয়া ভালো। চলো পার্কে যাই। আজ শুক্রবার। সুপ্রভাত। এই আমার বন্ধু। অনুগ্রহ করে আমাকে সাহায্য করুন। আপনি কি খেয়েছেন? আমি বাংলা শিখছি। শুভ রাত্রি।"
        elif language == "pl":
            yield "Lubię czytać. Pogoda jest ładna. Chodźmy do parku. Dziś jest piątek. Dzień dobry. To jest mój przyjaciel. Proszę mi pomóc. Czy jadłeś? Uczę się polskiego. Dobranoc."

    def synthesize(engine, language, generator):
        stream = TextToAudioStream(engine)

        print(f"Starting to play stream in {language}")
        stream.feed(generator(language))
        filename = f"synthesis_{language}_" + engine.engine_name

        tokenizer = (
            "stanza" if language in ["zh", "es", "de", "fr", "it", "ja", "ko", "ar", "hi", "pt", "ru", "id", "tr", "vi", "bn", "pl"] else None
        )
        stream.play(
            minimum_sentence_length=2,
            minimum_first_fragment_length=2,
            output_wavfile=f"{filename}.wav",
            on_sentence_synthesized=lambda sentence: print(
                f"Synthesized ({language}): " + sentence
            ),
            tokenizer=tokenizer,
            language=language,
            context_size=2,
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
                "de": "female_german",
                "fr": "female_french",
                "it": "female_italian",
                "ja": "female_japanese",
                "ko": "female_korean",
                "ar": "female_arabic",
                "hi": "female_hindi",
                "pt": "female_portuguese",
                "ru": "female_russian",
                "id": "female_indonesian",
                "tr": "female_turkish",
                "vi": "female_vietnamese",
                "bn": "female_bengali",
                "pl": "female_polish",
            }
            return CoquiEngine(voice=voices[language], language=language)
        elif name == "azure":
            from RealtimeTTS import AzureEngine

            voices = {
                "zh": "zh-CN-XiaoxiaoNeural",
                "en": "en-US-JennyNeural",
                "es": "es-ES-ElviraNeural",
                "de": "de-DE-KatjaNeural",
                "fr": "fr-FR-DeniseNeural",
                "it": "it-IT-ElsaNeural",
                "ja": "ja-JP-AoiNeural",
                "ko": "ko-KR-SunHiNeural",
                "ar": "ar-AE-FatimaNeural",
                "hi": "hi-IN-SwaraNeural",
                "pt": "pt-BR-FranciscaNeural",
                "ru": "ru-RU-SvetlanaNeural",
                "id": "id-ID-GadisNeural",
                "tr": "tr-TR-EmelNeural",
                "vi": "vi-VN-HoaiMyNeural",
                "bn": "bn-IN-TanishaaNeural",
                "pl": "pl-PL-AgnieszkaNeural",
            }
            return AzureEngine(
                os.environ.get("AZURE_SPEECH_KEY"),
                os.environ.get("AZURE_SPEECH_REGION"),
                voice=voices[language],
            )
        elif name == "elevenlabs":
            from RealtimeTTS import ElevenlabsEngine

            return ElevenlabsEngine(os.environ.get("ELEVENLABS_API_KEY"))
        else:
            from RealtimeTTS import SystemEngine

            voices = {
                "zh": "Huihui",
                "en": "Microsoft Zira",
                "es": "Microsoft Helena",
                "de": "Microsoft Hedda",
                "fr": "Microsoft Hortense",
                "it": "Microsoft Elsa",
                "ja": "Microsoft Haruka",
                "ko": "Microsoft Heami",
                "ar": "Microsoft Hoda",
                "hi": "Microsoft Hemant",
                "pt": "Microsoft Maria",
                "ru": "Microsoft Irina",
                "id": "Microsoft Andika",  # Note: This is a placeholder, as Windows might not have a default Indonesian voice
                "tr": "Microsoft Tolga",  # Note: This is a placeholder, as Windows might not have a default Turkish voice
                "vi": "Microsoft An",  # Note: This is a placeholder, as Windows might not have a default Vietnamese voice
                "bn": "Microsoft Bashkar",  # Note: This is a placeholder, as Windows might not have a default Bengali voice
                "pl": "Microsoft Paulina",  # Note: This is a placeholder, as Windows might not have a default Polish voice
            }
            return SystemEngine(voice=voices[language])

    languages = ["zh", "en", "es", "de", "fr", "it", "ja", "ko", "ar", "hi", "pt", "ru", "id", "tr", "vi", "bn", "pl"]

    for engine_name in ["coqui", "elevenlabs", "azure", "system"]:
        for language in languages:
            print(f"Starting engine: {engine_name} for language: {language}")
            engine = get_engine(engine_name, language)
            print(f"Synthesizing with engine: {engine_name} for language: {language}")
            synthesize(engine, language, dummy_generator)
