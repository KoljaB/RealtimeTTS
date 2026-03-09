"""
CAMB AI MARS TTS – Conversational AI Bot Demo

A multi-turn chat bot that streams LLM responses through CAMB AI's MARS
text-to-speech engine in real time.

Commands:
    /model mars-flash|mars-pro|mars-instruct   Switch TTS model
    /lang <bcp47-code>                          Switch language (en-us, es, fr, de, ja, zh …)
    /voice <id>                                 Switch voice by numeric ID
    /instruct <text>                            Set user instructions (mars-instruct only)
    /quit                                       Exit

Env vars: CAMB_API_KEY, OPENAI_API_KEY
"""

if __name__ == "__main__":
    import os
    import sys
    import time
    from dotenv import load_dotenv

    load_dotenv()

    from openai import OpenAI
    from RealtimeTTS import TextToAudioStream, CambEngine

    # ── configuration ──────────────────────────────────────────────────
    VALID_MODELS = {"mars-flash", "mars-pro", "mars-instruct"}
    DEFAULT_MODEL = "mars-flash"
    DEFAULT_LANG = "en-us"
    DEFAULT_VOICE = 147320

    # ── initialise ─────────────────────────────────────────────────────
    openai_client = OpenAI()
    engine = CambEngine(
        speech_model=DEFAULT_MODEL,
        language=DEFAULT_LANG,
        voice_id=DEFAULT_VOICE,
    )
    stream = TextToAudioStream(engine)

    current_model = DEFAULT_MODEL
    current_lang = DEFAULT_LANG
    current_voice = DEFAULT_VOICE
    user_instructions = None
    chat_history = []

    def build_system_prompt():
        lang_note = (
            f" Always respond in the language with BCP-47 code '{current_lang}'."
            if current_lang != "en-us"
            else ""
        )
        return (
            "You are a helpful, concise conversational assistant."
            " Keep answers to two or three sentences unless asked for more."
            f"{lang_note}"
        )

    def llm_stream(user_text):
        """Yield text chunks from the OpenAI chat completion stream."""
        chat_history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": build_system_prompt()}] + chat_history

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )

        full_reply = []
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                full_reply.append(delta.content)
                yield delta.content

        chat_history.append({"role": "assistant", "content": "".join(full_reply)})

    def speak_and_wait(text_or_generator):
        """Feed text/generator to the TTS stream and block until playback ends."""
        stream.feed(text_or_generator)
        stream.play_async()
        while stream.is_playing():
            time.sleep(0.1)

    # ── startup greeting ───────────────────────────────────────────────
    print("Initialising CAMB AI MARS TTS engine…")
    speak_and_wait("Hello! CAMB AI MARS engine is ready. Let's chat.")
    print()

    # ── help text ──────────────────────────────────────────────────────
    print("Commands:")
    print("  /model mars-flash|mars-pro|mars-instruct")
    print("  /lang  <bcp47-code>   (en-us, es, fr, de, ja, zh …)")
    print("  /voice <id>           (numeric voice ID)")
    print("  /instruct <text>      (set instructions for mars-instruct)")
    print("  /quit")
    print()

    # ── main loop ──────────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        # ── slash commands ─────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "/quit":
                break

            elif cmd == "/model":
                if arg not in VALID_MODELS:
                    print(f"Invalid model. Choose from: {', '.join(sorted(VALID_MODELS))}")
                    continue
                current_model = arg
                engine.set_voice_parameters(speech_model=current_model)
                print(f"Model set to {current_model}")

            elif cmd == "/lang":
                if not arg:
                    print("Usage: /lang <bcp47-code>")
                    continue
                current_lang = arg.lower()
                engine.set_voice_parameters(language=current_lang)
                print(f"Language set to {current_lang}")

            elif cmd == "/voice":
                if not arg.isdigit():
                    print("Usage: /voice <numeric-id>")
                    continue
                current_voice = int(arg)
                engine.set_voice(current_voice)
                print(f"Voice set to {current_voice}")

            elif cmd == "/instruct":
                if not arg:
                    print("Usage: /instruct <text>")
                    continue
                user_instructions = arg
                engine.set_voice_parameters(user_instructions=user_instructions)
                print(f"Instructions set: {user_instructions}")

            else:
                print(f"Unknown command: {cmd}")

            continue

        # ── chat with LLM → TTS ───────────────────────────────────────
        print("Assistant: ", end="", flush=True)
        speak_and_wait(llm_stream(user_input))
        print()

    # ── cleanup ────────────────────────────────────────────────────────
    stream.stop()
    engine.shutdown()
    print("Goodbye!")
