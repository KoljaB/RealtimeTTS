#!/usr/bin/env python
"""Simple Inflect-Micro-v2 playback test.

Install both backends:
    pip install "realtimetts[inflect]"

Or install only the backend you need:
    pip install "realtimetts[inflect-pytorch]"
    pip install "realtimetts[inflect-onnx]"
"""


if __name__ == "__main__":
    import sys

    # RealtimeTTS prints a few Unicode progress markers; keep the manual test
    # usable in the stock Windows cp1252 console as well as UTF-8 terminals.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from RealtimeTTS import InflectEngine, TextToAudioStream

    def text_generator():
        yield "Hello from the Inflect Micro version two engine. "
        yield "This is a simple realtime streaming test using Realtime TTS."

    # Auto uses PyTorch on CUDA, ONNX on CPU when installed, and otherwise
    # falls back to PyTorch CPU.
    engine = InflectEngine()
    stream = TextToAudioStream(engine)

    print(f"Using Inflect backend={engine.backend}, device={engine.device}")
    try:
        stream.feed(text_generator()).play(log_synthesized_text=True)
    finally:
        engine.shutdown()
