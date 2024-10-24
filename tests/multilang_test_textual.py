from textual.app import App, ComposeResult
from textual.widgets import Static, Select, TextArea, Button, LoadingIndicator
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.reactive import reactive
from rich.console import Console

from time import sleep
import os
import re
import threading
import asyncio

import sys
import queue
import threading
from contextlib import redirect_stdout, redirect_stderr

# Initialize the rich console for colored output
console = Console()

# Load all engines here
engine_names = ["coqui", "system"]

# Check environment variables for Azure and Elevenlabs
elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY")
azure_key = os.environ.get("AZURE_SPEECH_KEY")
azure_region = os.environ.get("AZURE_SPEECH_REGION")
openai_key = os.environ.get("OPENAI_API_KEY")

if elevenlabs_key:
    engine_names.append("elevenlabs")
else:
    console.print("[red]ELEVENLABS_API_KEY is missing, Elevenlabs engine will not be loaded.[/red]")

if azure_key and azure_region:
    engine_names.append("azure")
else:
    if not azure_key:
        console.print("[red]AZURE_SPEECH_KEY is missing, Azure engine will not be loaded.[/red]")
    if not azure_region:
        console.print("[red]AZURE_SPEECH_REGION is missing, Azure engine will not be loaded.[/red]")

if openai_key:
    engine_names.append("openai")
else:
    console.print("[red]OPENAI_API_KEY is missing, OpenAI engine will not be loaded.[/red]")

available_providers = engine_names

coqui_supported_languages = {
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "hu",
    "ko",
    "ja",
    "hi",
}

# Extracted language-specific configurations
languages = {
    "zh-cn": {
        "name": "Chinese",
        "default_text": "我喜欢读书。天气很好。我们去公园吧。今天是星期五。早上好。这是我的朋友。请帮我。吃饭了吗？我在学中文。晚安。",
        "voices": {
            "coqui": "female_chinese",
            "azure": "XiaoxiaoNeural",
            "system": "Huihui",
        },
        "remove_punctuation": False,
    },
    "en": {
        "name": "English",
        "default_text": "I love reading. The weather is nice. Let's go to the park. Today is Friday. Good morning. This is my friend. Please help me. Have you eaten? I'm learning English. Good night.",
        "voices": {
            "coqui": "female_english",
            "azure": "JennyNeural",
            "system": "Microsoft Zira",
        },
        "remove_punctuation": False,
    },
    "es": {
        "name": "Spanish",
        "default_text": "Me gusta leer. Hace buen tiempo. Vamos al parque. Hoy es viernes. Buenos días. Este es mi amigo. Por favor, ayúdame. ¿Has comido? Estoy aprendiendo español. Buenas noches.",
        "voices": {
            "coqui": "female_spanish",
            "azure": "ElviraNeural",
            "system": "Microsoft Helena",
        },
        "remove_punctuation": True,
    },
    "de": {
        "name": "German",
        "default_text": "Ich liebe es zu lesen. Das Wetter ist schön. Lass uns in den Park gehen. Heute ist Freitag. Guten Morgen. Das ist mein Freund. Bitte hilf mir. Hast du gegessen? Ich lerne Deutsch. Gute Nacht.",
        "voices": {
            "coqui": "female_german",
            "azure": "KatjaNeural",
            "system": "Microsoft Hedda",
        },
        "remove_punctuation": True,
    },
    "fr": {
        "name": "French",
        "default_text": "J'aime lire. Le temps est agréable. Allons au parc. Aujourd'hui, c'est vendredi. Bonjour. C'est mon ami. Aidez-moi s'il vous plaît. As-tu mangé? J'apprends le français. Bonne nuit.",
        "voices": {
            "coqui": "female_french",
            "azure": "DeniseNeural",
            "system": "Microsoft Hortense",
        },
        "remove_punctuation": True,
    },
    "it": {
        "name": "Italian",
        "default_text": "Mi piace leggere. Il tempo è bello. Andiamo al parco. Oggi è venerdì. Buongiorno. Questo è il mio amico. Per favore aiutami. Hai mangiato? Sto imparando l'italiano. Buonanotte.",
        "voices": {
            "coqui": "female_italian",
            "azure": "ElsaNeural",
            "system": "Microsoft Elsa",
        },
        "remove_punctuation": True,
    },
    "ar": {
        "name": "Arabic",
        "default_text": "أحب القراءة. الطقس لطيف. هيا نذهب إلى الحديقة. اليوم هو الجمعة. صباح الخير. هذا صديقي. أرجوك ساعدني. هل تناولت الطعام؟ أنا أتعلم العربية. تصبح على خير.",
        "voices": {
            "coqui": "female_arabic",
            "azure": "FatimaNeural",
            "system": "Microsoft Hoda",
        },
        "remove_punctuation": False,
    },
    "hi": {
        "name": "Hindi",
        "default_text": "मुझे पढ़ना पसंद है। मौसम अच्छा है। चलो पार्क चलते हैं। आज शुक्रवार है। सुप्रभात। यह मेरा दोस्त है। कृपया मेरी मदद करें। क्या आपने खाना खाया? मैं हिंदी सीख रहा हूं। शुभ रात्रि।",
        "voices": {
            "coqui": "female_hindi",
            "azure": "SwaraNeural",
            "system": "Microsoft Hemant",
        },
        "remove_punctuation": False,
    },
    "pt": {
        "name": "Portuguese",
        "default_text": "Eu gosto de ler. O tempo está bom. Vamos ao parque. Hoje é sexta-feira. Bom dia. Este é meu amigo. Por favor, me ajude. Você já comeu? Estou aprendendo português. Boa noite.",
        "voices": {
            "coqui": "female_portuguese",
            "azure": "FranciscaNeural",
            "system": "Microsoft Maria",
        },
        "remove_punctuation": True,
    },
    "ru": {
        "name": "Russian",
        "default_text": "Я люблю читать. Погода хорошая. Пойдем в парк. Сегодня пятница. Доброе утро. Это мой друг. Пожалуйста, помогите мне. Вы уже ели? Я учу русский язык. Спокойной ночи.",
        "voices": {
            "coqui": "female_russian",
            "azure": "SvetlanaNeural",
            "system": "Microsoft Irina",
        },
        "remove_punctuation": True,
    },
    "id": {
        "name": "Indonesian",
        "default_text": "Saya suka membaca. Cuacanya bagus. Ayo pergi ke taman. Hari ini hari Jumat. Selamat pagi. Ini teman saya. Tolong bantu saya. Apakah Anda sudah makan? Saya sedang belajar bahasa Indonesia. Selamat malam.",
        "voices": {
            "coqui": "female_indonesian",
            "azure": "GadisNeural",
            "system": "Microsoft Andika",  # Placeholder; may not exist
        },
        "remove_punctuation": False,
    },
    "tr": {
        "name": "Turkish",
        "default_text": "Okumayı seviyorum. Hava güzel. Haydi parka gidelim. Bugün Cuma. Günaydın. Bu benim arkadaşım. Lütfen bana yardım edin. Yemek yediniz mi? Türkçe öğreniyorum. İyi geceler.",
        "voices": {
            "coqui": "female_turkish",
            "azure": "EmelNeural",
            "system": "Microsoft Tolga",  # Placeholder; may not exist
        },
        "remove_punctuation": True,
    },
    "vi": {
        "name": "Vietnamese",
        "default_text": "Tôi thích đọc sách. Thời tiết đẹp. Hãy đi công viên. Hôm nay là thứ Sáu. Chào buổi sáng. Đây là bạn tôi. Làm ơn giúp tôi. Bạn đã ăn chưa? Tôi đang học tiếng Việt. Chúc ngủ ngon.",
        "voices": {
            "coqui": "female_vietnamese",
            "azure": "HoaiMyNeural",
            "system": "Microsoft An",  # Placeholder; may not exist
        },
        "remove_punctuation": False,
    },
    "bn": {
        "name": "Bengali",
        "default_text": "আমি পড়তে পছন্দ করি। আবহাওয়া ভালো। চলো পার্কে যাই। আজ শুক্রবার। সুপ্রভাত। এই আমার বন্ধু। অনুগ্রহ করে আমাকে সাহায্য করুন। আপনি কি খেয়েছেন? আমি বাংলা শিখছি। শুভ রাত্রি।",
        "voices": {
            "coqui": "female_bengali",
            "azure": "TanishaaNeural",
            "system": "Microsoft Bashkar",  # Placeholder; may not exist
        },
        "remove_punctuation": False,
    },
    "cs": {
        "name": "Czech",
        "default_text": "Rád čtu. Počasí je hezké. Pojďme do parku. Dnes je pátek. Dobré ráno. Tohle je můj přítel. Prosím, pomoz mi. Jedl jsi? Učím se česky. Dobrou noc.",
        "voices": {
            "coqui": "female_czech",
            "azure": "VlastaNeural",
            "system": "Microsoft Zuzana",
        },
        "remove_punctuation": True,
    },
    "hu": {
        "name": "Hungarian",
        "default_text": "Szeretek olvasni. Szép az idő. Menjünk a parkba. Ma péntek van. Jó reggelt. Ez a barátom. Kérlek, segíts. Ettél már? Magyarul tanulok. Jó éjszakát.",
        "voices": {
            "coqui": "female_hungarian",
            "azure": "NoemiNeural",
            "system": "Microsoft Szabolcs",
        },
        "remove_punctuation": True,
    },
    "nl": {
        "name": "Dutch",
        "default_text": "Ik hou van lezen. Het weer is mooi. Laten we naar het park gaan. Vandaag is het vrijdag. Goedemorgen. Dit is mijn vriend. Help me alstublieft. Heb je gegeten? Ik leer Nederlands. Welterusten.",
        "voices": {
            "coqui": "female_dutch",
            "azure": "ColetteNeural",
            "system": "Microsoft Frank",
        },
        "remove_punctuation": True,
    },
    "pl": {
        "name": "Polish",
        "default_text": "Lubię czytać. Pogoda jest ładna. Chodźmy do parku. Dziś jest piątek. Dzień dobry. To mój przyjaciel. Proszę, pomóż mi. Czy jadłeś? Uczę się polskiego. Dobranoc.",
        "voices": {
            "coqui": "female_polish",
            "azure": "AgnieszkaNeural",
            "system": "Microsoft Paulina",
        },
        "remove_punctuation": True,
    },    
    "ko": {
        "name": "Korean",
        "default_text": "저는 독서를 좋아해요. 날씨가 좋네요. 공원에 가요. 오늘은 금요일이에요. 좋은 아침이에요. 이 분은 제 친구예요. 제발 도와주세요. 식사하셨어요? 저는 한국어를 배우고 있어요. 안녕히 주무세요.",
        "voices": {
            "coqui": "female_korean",
            "azure": "SunHiNeural",
            "system": "Microsoft Heami",
        },
        "remove_punctuation": True,
    },
    "ja": {
        "name": "Japanese",
        "default_text": "私は読書が好きです。天気がいいですね。公園に行きましょう。今日は金曜日です。おはようございます。これは私の友達です。助けてください。食べましたか？私は日本語を勉強しています。おやすみなさい。",
        "voices": {
            "coqui": "female_japanese",
            "azure": "AoiNeural",
            "system": "Microsoft Haruka",
        },
        "remove_punctuation": False,
    },
}

class QueueWriter:
    def __init__(self, queue):
        self.queue = queue

    def write(self, message):
        # Ensure message is a string
        self.queue.put(str(message))

    def flush(self):
        pass  # No need to implement flush in this case

class LoadingScreen(Screen):
    """A screen that displays a loading message and logs."""

    CSS = """
    #loading_message {
        text-align: center;
        width: 100%;
        padding: 2; /* Adjust this as needed */
    }
    #loading_indicator {
        align: center middle;
        margin-top: 2; /* Adjust the margin-top to ensure visibility */
        height: 3; /* Explicit height for visibility */
    }
    #log_label {
        margin: 1 1 0 1;
        text-align: left;
        padding: 1;
        width: 100%;
        color: $primary;
    }    
    #log_output {
        color: $secondary;    
        margin: 0 1 0 1;
        padding: 0 2 0 2;    
        overflow-y: auto;
        height: 40%; /* Adjust height for better layout */
        width: 100%;
    }
    Screen {
        overflow-y: hidden;  /* Disable vertical scrolling */
        height: 100%; /* Make sure screen takes full height */
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("... Loading Text-to-Speech Engines ...", classes="loading_message", id="loading_message")
            yield Static("Engine startup logging:", id="log_label")
            yield TextArea("", id="log_output", read_only=True)
            yield LoadingIndicator(id="loading_indicator")

    def on_mount(self):
        # Center the loading message and indicator
        self.set_interval(0.5, self.animate_loading_message)
        # Set up a periodic task to update the log output
        self.set_interval(0.5, self.update_log_output)

    def animate_loading_message(self):
        loading_message = self.query_one("#loading_message", Static)
        if loading_message:
            text = str(loading_message.renderable).rstrip(".")
            dot_count = (text.count('.') + 1) % 4
            loading_message.update(f"{'.' * dot_count} Loading Text-to-Speech Engines {'.' * dot_count}")

    def update_log_output(self):
        # Read from the log queue and update the log_output TextArea
        log_output = self.query_one("#log_output", TextArea)
        if log_output:
            log_content = log_output.text  # Get the current content
            new_content = False  # Track if new content is added

            while not self.app.log_queue.empty():
                message = self.app.log_queue.get()
                log_content += message + "\n"  # Append the new message
                new_content = True  # Mark that new content was added

            if new_content:
                log_output.text = log_content  # Update the TextArea content
                log_output.scroll_end()  # Automatically scroll to the bottom

class BasicApp(App):
    CSS = """
    Screen {
        background: black;
    }
    Static.description {
        margin: 1 1 0 2;
        color: $secondary;
        text-align: left;
    }
    Select#options {
        margin: 0 0;
        width: 50%;
        padding: 1;
        align: center middle;
    }
    Select#language {
        margin: 0 0;
        width: 50%;
        padding: 1;
        align: center middle;
    }
    TextArea#text_to_speak {
        margin: 1 1 0 1;
        padding: 0 2 0 2;
    }  
    Button {
        width: 20%;
        margin: 2 2;
    }
    .loading_message {
        text-align: center;
        margin-top: 2;
        color: $primary;
    }
    #loading_indicator {
        margin-top: 1;
        align: center middle;
    }
    """

    engines = reactive({})
    available_providers = reactive([])

    def __init__(self):
        super().__init__()
        self.stream = None
        self.engine = None
        self.is_coqui = False
        self.current_language = None
        self.log_queue = queue.Queue()

    def _log(self, message):
        """Helper function to capture and display logs."""
        self.loading_logs.write(message + "\n")
        self.loading_screen.append_log(message)
        console.print(message)  # Also print it to the terminal

    async def on_mount(self):
        # Show the loading screen
        await self.push_screen(LoadingScreen())

        # Start loading engines in the background
        asyncio.create_task(self.load_engines())

    async def load_engines(self):
        # Load the engines in a background thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_engines_sync)

        # After engines are loaded, pop the loading screen
        await self.pop_screen()

        # Now that engines are loaded, recompose the app to reflect available options
        self.refresh()

    def _load_engines_sync(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = QueueWriter(self.log_queue)
        sys.stderr = QueueWriter(self.log_queue)
        try:
            self.engines = {}

            # Get the selected language (default to 'en' if not selected yet)
            language = 'en'

            for name in engine_names:
                engine = self.create_engine(name, language)  # Default to English for now
                if engine:
                    self.engines[name] = engine
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def create_engine(self, name, language):
        try:
            voice = languages.get(language, {}).get("voices", {}).get(name)
            if name == "coqui":
                from RealtimeTTS import CoquiEngine, TextToAudioStream

                # Create the Coqui engine
                engine = CoquiEngine(
                    voice=voice or "female_english", 
                    language=language,
                    add_sentence_filter=False,
                )

                # Wake up the Coqui engine after creation
                stream = TextToAudioStream(engine)
                # Use default text for wakeup
                default_text = languages.get(language, {}).get("default_text", "Hello")

                stream.feed([default_text])

                tokenizer = "stanza" if language in ["zh", "ja", "ko"] else None
                sentence_length = 2 if language in ["zh", "ja", "ko"] else 7

                stream.play(
                    minimum_sentence_length=sentence_length,
                    minimum_first_fragment_length=sentence_length,
                    tokenizer=tokenizer,
                    language=language,
                    context_size=sentence_length,
                    muted=True  # Ensure the wakeup is silent
                )

                # Clean up the stream
                stream = None

                return engine

            elif name == "azure":
                from RealtimeTTS import AzureEngine

                return AzureEngine(
                    os.environ.get("AZURE_SPEECH_KEY"),
                    os.environ.get("AZURE_SPEECH_REGION"),
                    voice=voice or "en-US-JennyNeural",
                )
            elif name == "elevenlabs":
                from RealtimeTTS import ElevenlabsEngine

                return ElevenlabsEngine(os.environ.get("ELEVENLABS_API_KEY"))
            elif name == "openai":
                from RealtimeTTS import OpenAIEngine

                return OpenAIEngine()
            else:
                from RealtimeTTS import SystemEngine

                return SystemEngine(voice=voice or "Microsoft Zira")
        except Exception as e:
            print(f"Failed to load engine {name}: {e}")
            return None
    def compose(self) -> ComposeResult:
        yield Static("Please choose a TTS provider:", classes="description")

        # Create Select with available providers
        if available_providers:
            yield Select(
                options=[(provider.capitalize(), provider) for provider in available_providers],
                id="options",
                prompt="Engine"
            )
        else:
            yield Static("No TTS providers available", classes="description")

        yield Static("Please choose a language:", classes="description")
        yield Select(
            options=[(lang_info["name"], code) for code, lang_info in languages.items()],
            id="language",
            prompt="Language"
        )

        yield Static("Text to speak:", classes="description")
        yield TextArea(id="text_to_speak")

        # Add the container for buttons
        yield Horizontal(
            Button("Speak", id="speak"),
            Button("Stop", id="pause"),
            Button("Exit", id="abort"),
        )

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "speak":
            self.speak()
        elif event.button.id == "pause":
            self.pause()
        elif event.button.id == "abort":
            self.abort()

    def on_select_changed(self, event: Select.Changed):
        if event.select.id == "options":
            self.pause()
            # Get the selected engine
            selected_engine = event.value

            # Filter out unsupported languages if "coqui" is selected
            language_options = [(lang_info["name"], code) for code, lang_info in languages.items()]
            self.is_coqui = selected_engine == "coqui"
            if selected_engine == "coqui":
                # Only include languages supported by Coqui
                language_options = [(name, code) for name, code in language_options if code in coqui_supported_languages]

            # Update the language select options dynamically
            language_select = self.query_one("#language", Select)
            language_select.set_options(language_options)

            # Refresh the app to reflect the new language options
            self.refresh()
     
        if event.select.id == "language":
            self.pause()
            new_language = event.value
            self.update_engines_language(new_language)
            
            # Prepopulate the text area with default text
            text_area = self.query_one("#text_to_speak", TextArea)
            default_text = languages.get(new_language, {}).get("default_text", "")
            text_area.text = default_text

    def update_engines_language(self, language):
        # Update the voice for each engine
        for name, engine in self.engines.items():
            voice = languages.get(language, {}).get("voices", {}).get(name)
            if voice:
                engine.set_voice(voice)
                if hasattr(engine, 'language'):
                    engine.language = language

        # If the engine is Coqui, adjust the add_sentence_filter based on remove_punctuation
        engine_select = self.query_one("#options", Select)
        engine_name = engine_select.value

        if engine_name == "coqui":
            engine = self.engines.get(engine_name)
            remove_punctuation = languages.get(language, {}).get("remove_punctuation", False)
            engine.add_sentence_filter = remove_punctuation                    

    def speak(self):
        if hasattr(self, 'stream') and self.stream is not None:
            print("Synthesis is already in progress.")
            return

        # Get the selected TTS provider
        engine_select = self.query_one("#options", Select)
        engine_name = engine_select.value

        # Get the selected language
        language_select = self.query_one("#language", Select)
        language = language_select.value

        # Get the text to speak
        text_area = self.query_one("#text_to_speak", TextArea)
        text_to_speak = text_area.text.strip()

        if not text_to_speak:
            print("Please enter text to speak.")
            return

        # Get the engine from the loaded engines
        engine = self.engines.get(engine_name)
        if not engine:
            print(f"Engine {engine_name} is not available.")
            return

        # Run the synthesis in a separate thread
        t = threading.Thread(target=self.synthesize_text, args=(engine, language, text_to_speak))
        t.start()

    def synthesize_text(self, engine, language, text):
        from RealtimeTTS import TextToAudioStream

        self.stream = TextToAudioStream(engine)
        self.engine = engine

        print(f"Starting to play stream in {language}")
        self.stream.feed([text])

        tokenizer = "stanza" if language in ["zh", "ja", "ko"] else None
        sentence_length = 2 if language in ["zh", "ja", "ko"] else 7

        self.current_language = language

        self.stream.play(
            minimum_sentence_length=sentence_length,
            minimum_first_fragment_length=sentence_length,
            tokenizer=tokenizer,
            language=language,
            context_size=sentence_length,            
            force_first_fragment_after_words=9999999
        )

        self.stream = None
        self.engine = None

    def pause(self):
        if hasattr(self, 'stream') and self.stream is not None:
            if self.stream.is_playing():
                self.stream.stop()
                sleep(0.5)

    def abort(self):
        exit()


if __name__ == "__main__":
    app = BasicApp()
    app.run()

    sleep(1)

    # Function to print the shutdown message, then update after shutdown completes
    def shutdown_engine(engine_name, engine):
        with console.status(f"[yellow]Shutting down {engine_name} engine...[/yellow]", spinner="dots") as status:
            if hasattr(engine, "shutdown"):
                engine.shutdown()  # Simulate engine shutdown process
                sleep(0.5)  # Simulate some delay for the shutdown process
            status.update(f"[green]✅ {engine_name} engine stopped.[/green]", spinner=None)
            sleep(0.5)  # Small delay to show the success message

    # Shutdown all engines
    def shutdown_all_engines(app):
        console.print("[bold red]Shutting down TTS engines...[/bold red]")
        for engine_name, engine in app.engines.items():
            shutdown_engine(engine_name, engine)
        
        console.print("[green]✅ All engines stopped.[/green]")
        console.print("[bold green]Exiting... Goodbye![/bold green]")

    shutdown_all_engines(app)
