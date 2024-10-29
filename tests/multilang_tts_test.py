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
        elif language == "nl":
            yield "Ik hou van lezen. Het weer is mooi. Laten we naar het park gaan. Vandaag is het vrijdag. Goedemorgen. Dit is mijn vriend. Help me alsjeblieft. Heb je al gegeten? Ik leer Nederlands. Welterusten."
        elif language == "uk":
            yield "Я люблю читати. Погода гарна. Давайте підемо в парк. Сьогодні п'ятниця. Доброго ранку. Це мій друг. Будь ласка, допоможіть мені. Ви вже їли? Я вивчаю українську. На добраніч."
        elif language == "be":
            yield "Я люблю чытаць. Надвор'е добрае. Пойдзем у парк. Сёння пятніца. Добрай раніцы. Гэта мой сябар. Калі ласка, дапамажыце мне. Вы ўжо елі? Я вывучаю беларускую мову. Дабранач."
        elif language == "ur":
            yield "مجھے پڑھنا پسند ہے۔ موسم اچھا ہے۔ آئیے پارک چلتے ہیں۔ آج جمعہ ہے۔ صبح بخیر۔ یہ میرا دوست ہے۔ براہ کرم مدد کیجیے۔ کیا آپ نے کھانا کھایا ہے؟ میں اردو سیکھ رہا ہوں۔ شب بخیر۔"
        elif language == "mr":
            yield "मला वाचायला आवडते. हवामान छान आहे. चला बागेत जाऊया. आज शुक्रवार आहे. शुभ प्रभात. हा माझा मित्र आहे. कृपया मला मदत करा. तुम्ही जेवण केले आहे का? मी मराठी शिकत आहे. शुभ रात्री."
        elif language == "te":
            yield "నాకు చదవడం ఇష్టం. వాతావరణం బాగుంది. పార్కుకి వెళ్దాం. ఈరోజు శుక્రవారం. శుభోదయం. ఇతను నా స్నేహితుడు. దయచేసి నాకు సహాయం చేయండి. మీరు భోజనం చేశారా? నేను తెలుగు నేర్చుకుంటున్నాను. శుభరాత്రి."
        elif language == "ta":
            yield "எனக்கு படிக்க பிடிக்கும். வானிலை நன்றாக உள்ளது. பூங்காவிற்கு செல்வோம். இன்று வெள்ளிக்கிழமை. காலை வணக்கம். இவர் என் நண்பர். தயவுசெய்து எனக்கு உதவுங்். நீங்கள சாப்பிட்டீர்களா? நான் தமிழ் கற்றுக்கொண்டிருக்கிறேன். இனிய இரவு வணக்கம்."
        elif language == "gu":
            yield "મને વાંચવું ગમે છે. હવામાન સરસ છે. ચાલો પાર્કમાં જઈએ. આજે શુક્રવાર છે. સુપ્રભાત. આ મારો મિત્ર છે. કૃપા કરીને મને મદદ કરો. તમે ખાધું? હું ગુજરાતી શીખી રહ્યો છું. શુભ રાત્રી."
        elif language == "cs":
            yield "Rád čtu. Počasí je hezké. Pojďme do parku. Dnes je pátek. Dobré ráno. Toto je můj přítel. Prosím, pomozte mi. Už jste jedli? Učím se česky. Dobrou noc."
        elif language == "hu":
            yield "Szeretek olvasni. Az idő szép. Menjünk a parkba. Ma péntek van. Jó reggelt. Ő a barátom. Kérem, segítsen nekem. Ettél már? Magyarul tanulok. Jó éjszakát."
        elif language == "ml":
            yield "എനിക്ക് വായിക്കാൻ ഇഷ്ടമാണ്. കാലാവസ്ഥ നല്ലതാണ്. നമുക്ക് പാർക്കിലേക്ക് പോകാം. ഇന്ന് വെള്ളിയാഴ്ചയാണ്. സുപ്രഭാതം. ഇദ്ദേഹം എന്റെ സുഹൃത്താണ്. ദയവായി എന്നെ സഹായിക്കൂ. നിങ്ങൾ ഭക്ഷണം കഴിച്ചോ? ഞാൻ മലയാളം പഠിക്കുകയാണ്. ശുഭരാത്രി."
        elif language == "kn":
            yield "ನನಗೆ ಓದಲು ಇಷ್ಟ. ಹವಾಮಾನ ಚೆನ್ನಾಗಿದೆ. ಉದ್ಯಾನವನಕ್ಕೆ ಹೋಗೋಣ. ಇಂದು ಶುಕ್ರವಾರ. ಶುಭೋದಯ. ಇವರು ನನ್ನ ಸ್ನೇಹಿತರು. ದಯವಿಟ್ಟು ನನಗೆ ಸಹಾಯ ಮಾಡಿ. ನೀವು ಊಟ ಮಾಡಿದ್ದೀರಾ? ನಾನು ಕನ್ನಡ ಕಲಿಯುತ್ತಿದ್ದೇನೆ. ಶುಭ ರಾತ್ರಿ."
        elif language == "ne":
            yield "मलाई पढ्न मन पर्छ। मौसम राम्रो छ। पार्कमा जाऔं। आज शुक्रबार हो। शुभ प्रभात। यो मेरो साथी हो। कृपया मलाई मद्दत गर्नुहोस्। के तपाईंले खाना खानुभयो? म नेपाली सिक्दैछु। शुभ रात्री।"
        elif language == "th":
            yield "ฉันชอบอ่านหนังสือ อากาศดี ไปที่สวนสาธารณะกัน วันนี้วันศุกร์ สวัสดีตอนเช้า นี่คือเพื่อนของฉัน กรุณาช่วยฉันด้วย คุณทานอาหารหรือยัง ฉันกำลังเรียนภาษาไทย ราตรีสวัสดิ์"

    def synthesize(engine, language, generator):
        stream = TextToAudioStream(engine)

        print(f"Starting to play stream in {language}")
        stream.feed(generator(language))
        filename = f"synthesis_{language}_" + engine.engine_name

        tokenizer = (
            tokenizer = "stanza" if language in ["zh", "es", "de", "fr", "it", "ja", "ko", "ar", "hi", "pt", "ru", "id", "tr", "vi", "bn", "pl", "nl", "uk", "be", "ur", "mr", "te", "ta", "gu", "cs", "hu", "ml", "kn", "ne", "th"] else None
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
                "nl": "female_dutch",
                "uk": "female_ukrainian",
                "be": "female_belarusian",  # Add this line
                "ur": "female_urdu",  # Add this line
                "mr": "female_marathi",  # Add this line
                "te": "female_telugu",  # Add this line
                "ta": "female_tamil",  # Add this line
                "gu": "female_gujarati",  # Add this line
                "cs": "female_czech",  # Add this line
                "hu": "female_hungarian",  # Add this line
                "ml": "female_malayalam",  # Add this line
                "kn": "female_kannada",  # Add this line
                "ne": "female_nepali",  # Add this line
                "th": "female_thai", # Add this line
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
                "nl": "nl-NL-ColetteNeural",
                "uk": "uk-UA-PolinaNeural",
                "be": "be-BY-DaryaNeural",  # Add this line
                "ur": "ur-PK-UzmaNeural",  # Add this line
                "mr": "mr-IN-AarohiNeural",  # Add this line
                "te": "te-IN-ShrutiNeural",  # Add this line
                "ta": "ta-IN-PallaviNeural",  # Add this line
                "gu": "gu-IN-DhwaniNeural",  # Add this line
                "cs": "cs-CZ-VlastaNeural",  # Add this line
                "hu": "hu-HU-NoemiNeural",  # Add this line
                "ml": "ml-IN-SobhanaNeural",  # Add this line
                "kn": "kn-IN-SapnaNeural",  # Add this line
                "ne": "ne-NP-HemkalaNeural",  # Add this line
                "th": "th-TH-PremwadeeNeural"  # Add this line
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
                "nl": "Microsoft Frank",  # Note: This is a placeholder, as Windows might not have a default Dutch voice
                "uk": "Microsoft Ostap",  # Note: This is a placeholder, as Windows might not have a default Ukrainian voice
                "be": "Microsoft Alena",  # Note: This is a placeholder, as Windows might not have a default Belarusian voice
                "ur": "Microsoft Asad",  # Note: This is a placeholder, as Windows might not have a default Urdu voice
                "mr": "Microsoft Swara",  # Note: This is a placeholder, as Windows might not have a default Marathi voice
                "te": "Microsoft Karthik",  # Note: This is a placeholder, as Windows might not have a default Telugu voice
                "ta": "Microsoft Valluvar",  # Note: This is a placeholder, as Windows might not have a default Tamil voice
                "gu": "Microsoft Dhwani",  # Note: This is a placeholder, as Windows might not have a default Gujarati voice
                "cs": "Microsoft Jakub",  # Note: This is a placeholder, as Windows might not have a default Czech voice
                "hu": "Microsoft Szabolcs",  # Note: This is a placeholder, as Windows might not have a default Hungarian voice
                "ml": "Microsoft Anjali",  # Note: This is a placeholder, as Windows might not have a default Malayalam voice
                "kn": "Microsoft Heera",  # Note: This is a placeholder, as Windows might not have a default Kannada voice
                "ne": "Microsoft Hemkala",  # Note: This is a placeholder, as Windows might not have a default Nepali voice
                "th": "Microsoft Premwadee",    # Note: This is a placeholder, as Windows might not have a default Thai voice
            }
            return SystemEngine(voice=voices[language])

    languages = ["zh", "en", "es", "de", "fr", "it", "ja", "ko", "ar", "hi", "pt", "ru", "id", "tr", "vi", "bn", "pl", "nl", "uk", "be", "ur", "mr", "te", "ta", "gu", "cs", "hu", "ml", "kn", "ne", "th"]

    for engine_name in ["coqui", "elevenlabs", "azure", "system"]:
        for language in languages:
            print(f"Starting engine: {engine_name} for language: {language}")
            engine = get_engine(engine_name, language)
            print(f"Synthesizing with engine: {engine_name} for language: {language}")
            synthesize(engine, language, dummy_generator)

