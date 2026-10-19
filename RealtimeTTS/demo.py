"""Mixed English/Spanish tutor demo. Run with python -m RealtimeTTS.demo."""
import argparse
from pathlib import Path

INSTRUCTIONS = {
    "normal": "Speak naturally at a conversational pace.",
    "surprised": "Sound surprised and amazed!",
    "whispered": "Whispering very quietly.",
    "pleased": "Sound pleased and encouraging.",
}

VOCALIZATIONS = {
    "laugh": "Brief laughter.",
    "clear_throat": "Clear your throat once, briefly.",
    "sigh": "Exhale a gentle sigh.",
}

TEXT = """
<en speed=1.1>
[normal]Welcome back! [clear_throat]
[normal]Let us take a little walk in Spanish.
[surprised]That is an absolutely enormous tree!
[normal]In Spanish, that tree is <es normal>El árbol.</es>
[normal]The trick is to practice a little every day.
[whispered]We should keep this little secret between you and me. [laugh]
[normal]Listen carefully while we say this sentence slowly: <es slow>Vamos al parque.</es>
[normal]Now try the same sentence at a normal pace: <es normal>Vamos al parque.</es>
[normal]Let us say a longer sentence a little faster: <es fast>Vamos al parque esta tarde.</es>
[normal]Let's take a deep breath now. [pause][sigh][pause]
[pleased]There we go! One last sentence before we head home: <es normal>El árbol está en el parque.</es>
[normal]Well done. A little practice really does go a long way.
</en>
"""



def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breeze-url", default="http://127.0.0.1:18084")
    parser.add_argument("--qwen-url", default="http://127.0.0.1:8080")
    parser.add_argument("--english-voice", default="breeze-en")
    parser.add_argument("--spanish-voice", default="qwen-es")
    parser.add_argument("--alignment-model", type=Path)
    parser.add_argument("--plain", action="store_true", help="Hide control tags in the display")
    parser.add_argument("--no-highlight", action="store_true", help="Play without the alignment model")
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    from contextlib import ExitStack
    from RealtimeTTS import LanguageRoutingEngine, RemoteTTSEngine, TextToAudioStream
    print("Connecting to Breeze and Qwen...", flush=True)
    with ExitStack() as cleanup:
        english = RemoteTTSEngine(args.breeze_url, args.english_voice, language="english")
        cleanup.callback(english.shutdown)
        spanish = RemoteTTSEngine(args.qwen_url, args.spanish_voice, language="spanish",
                                  seed=42, max_new_tokens=256)
        cleanup.callback(spanish.shutdown)
        engine = LanguageRoutingEngine({"en": english, "*": spanish}, pre_first_pcm_timeout=45)
        cleanup.callback(engine.shutdown)
        display = None
        options = {}
        if not args.no_highlight:
            from .terminal_highlighting import TerminalHighlighter
            from .model_assets import alignment_model
            display = TerminalHighlighter(args.alignment_model or alignment_model(),
                                          input_text=None if args.plain else TEXT)
            options = dict(muted=True, alignment=display.alignment, on_alignment=display.update)
        with TextToAudioStream(engine, language_tags=True, **options) as stream:
            stream.add_instructions(INSTRUCTIONS)
            stream.add_vocalizations(VOCALIZATIONS)
            stream.add_pause(duration=0.7)
            stream.feed(TEXT)
            if display:
                display.play(stream)
            else:
                print(TEXT)
                stream.play()


if __name__ == "__main__":
    main()
