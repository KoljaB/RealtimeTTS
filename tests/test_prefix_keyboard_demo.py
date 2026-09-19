import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading

spec = importlib.util.spec_from_file_location("keyboard_demo", Path(__file__).parents[1]/"tools/prefix_splice_keyboard_demo.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_whole_text_is_submitted_and_ttft_starts_at_keypress():
    received=[]
    class Stream:
        def __init__(self, engine, on_audio_stream_start): self.callback=on_audio_stream_start
        def feed(self, text): received.append(text); return self
        def play(self): self.callback()
    demo=module.KeyboardDemo(SimpleNamespace(metrics=[]), "Heute Morgen gehen wir.", Stream, clock=lambda: 12.25)
    assert demo.start(12.0)
    demo.worker.join(1)
    assert received == ["Heute Morgen gehen wir."]
    assert demo.ttft_ms == 250
    assert "250.0 ms" in demo.messages.get_nowait()


def test_repeat_press_does_not_reset_running_measurement_and_stop_works():
    entered=threading.Event(); release=threading.Event()
    class Stream:
        def __init__(self, engine, on_audio_stream_start): pass
        def feed(self, text): return self
        def play(self): entered.set(); release.wait(2)
        def stop(self): release.set()
    demo=module.KeyboardDemo(SimpleNamespace(metrics=[]), "Heute Morgen.", Stream)
    assert demo.start(1)
    assert entered.wait(1)
    assert not demo.start(2)
    assert demo.pressed_at == 1
    demo.stop()
    assert not demo.worker.is_alive()
