#!/usr/bin/env python
"""
First install:
    pip install "RealtimeTTS[all,jp,zh]"

Then install torch with CUDA support:
    pip install torch==2.5.1+cu121 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    (adjust 121 to your CUDA version, this is for CUDA 12.1, for CUDA 11.8 use 118)

"""
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

#engine = KokoroEngine(default_voice=languages["a"][0], debug=True)
engine = KokoroEngine(default_voice=languages["a"][0])

for lang, (voice, text) in prewarm_texts.items():
    print(f"Prewarming {voice} ({lang})")
    engine.set_voice(voice)
    TextToAudioStream(engine).feed([text]).play(muted=True)


import random
for lang, (voice, text) in languages.items():
    engine.set_voice(voice)
    # Generate speed between 0.7 and 1.8 (1.0 ± [−0.3, +0.8])
    speed = max(0.1, 1.0 + random.uniform(-0.3, 0.8))
     
    engine.set_voice(voice)
    engine.set_speed(speed)

    print(f"Testing {voice} ({lang}) using speed: {speed:.2f}")
    TextToAudioStream(engine).feed([text]).play(log_synthesized_text=True)

engine.shutdown()
