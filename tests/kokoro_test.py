#!/usr/bin/env python 
"""
First install:
    pip install "RealtimeTTS[all,jp,zh]"

Then install torch with CUDA support:
    pip install torch==2.5.1+cu121 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    (adjust 121 to your CUDA version, this is for CUDA 12.1, for CUDA 11.8 use 118)
"""
import string
import os
import time
import sys
import random
from RealtimeTTS import TextToAudioStream, KokoroEngine

languages = {
    "a": ("af_heart", "Hello, this is an American voice test."),
    "b": ("bf_emma", "Good day, mate! This is a British voice test."),
    "j": ("jf_alpha", "こんにちは、これは日本語のテストです。"),
    "z": ("zf_xiaobei", "你好，这是一段中文测试。"),
    "e": ("ef_dora", "¡Hola! Esta es una prueba de voz en español."),
    "f": ("ff_siwis", "Bonjour, ceci est un test de voix en français."),
    "h": ("hf_alpha", "नमस्ते, यह हिंदी में एक वॉयस टेस्ट है।"),
    "i": ("if_sara", "Ciao, questo è un test vocale in italiano."),
    "p": ("pf_dora", "Olá, este é um teste de voz em português brasileiro.")
}

prewarm_texts = {
    "a": ("af_heart", "Warm up"),
    "b": ("bf_emma", "Warm up"),
    "j": ("jf_alpha", "準備中"),
    "z": ("zf_xiaobei", "预热"),
    "e": ("ef_dora", "Preparando"),
    "f": ("ff_siwis", "Préchauffage"),
    "h": ("hf_alpha", "तैयारी"),
    "i": ("if_sara", "Riscaldamento"),
    "p": ("pf_dora", "Aquecendo")
}

engine = KokoroEngine()

# Prewarm voices
stream = TextToAudioStream(engine)
for lang, (voice, text) in prewarm_texts.items():
    print(f"Prewarming {voice} ({lang})")
    engine.set_voice(voice)
    stream.feed([text]).play(muted=True)

# Clear the screen in a platform-independent way
if sys.platform.startswith("win"):
    os.system("cls")
else:
    os.system("clear")

last_word = None

def process_word(word):
    global last_word
    if last_word and word.word not in set(string.punctuation):
        print(" ", end="", flush=True)
    print(f"{word.word}", end="", flush=True)
    last_word = word.word

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

start_time = 0
def on_audio_stream_start_callback():
    global start_time
    delta = time.time() - start_time
    print("<TTFT>", f"Time: {delta:.2f}s")

stream = TextToAudioStream(
    engine,
    log_characters=True,
    on_word=process_word,
    on_audio_stream_start=on_audio_stream_start_callback,
)
for lang, (voice, text) in languages.items():
    # Generate a random speed between 0.6 and 1.8 (1.0 ± [−0.4, +0.8])
    speed = max(0.1, 1.0 + random.uniform(-0.4, 0.8))
    # For testing, setting a fixed speed
    speed = 0.7

    engine.set_voice(voice)
    engine.set_speed(speed)

    if last_word:
        print()

    print(f"Testing {voice} ({lang}) using speed: {speed:.2f}")

    last_word = None
    start_time = time.time()
    before_sentence_callback, on_sentence_callback = create_synthesis_callbacks(start_time)

    stream.feed(text).play(
        before_sentence_synthesized=before_sentence_callback,
        on_sentence_synthesized=on_sentence_callback
    )

engine.shutdown()
