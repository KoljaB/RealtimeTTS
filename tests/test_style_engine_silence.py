import importlib
import sys
import threading
import types
from queue import Queue

import numpy as np


def _load_style_engine(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.delitem(sys.modules, "RealtimeTTS.engines.style_engine", raising=False)
    return importlib.import_module("RealtimeTTS.engines.style_engine")


def test_styletts_pause_matches_its_int16_stream_format(monkeypatch):
    module = _load_style_engine(monkeypatch)
    engine = object.__new__(module.StyleTTSEngine)
    engine.queue = Queue()
    engine.stop_synthesis_event = threading.Event()
    engine.trim_silence = False
    engine.alpha = engine.beta = engine.diffusion_steps = engine.embedding_scale = None
    engine.sentence_silence_duration = 0.6
    engine.inference = lambda text, **kwargs: np.zeros(2400, dtype=np.float32)

    assert engine.synthesize("Hello.")

    engine.queue.get()
    pause = engine.queue.get()
    # get_stream_info() reports paInt16, so 0.6 s at 24 kHz is 14400 two-byte samples.
    assert len(pause) == int(24000 * 0.6) * 2
