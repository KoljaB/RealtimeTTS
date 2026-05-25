import numpy as np
import sys
import types

from RealtimeTTS.engines.pocket_engine import PocketTTSEngine, PocketTTSVoice


def test_pocket_engine_streams_audio_on_worker(monkeypatch):
    class FakeModel:
        sample_rate = 24000

        def generate_audio_stream(self, voice_state, text, **kwargs):
            assert voice_state == "fake-state"
            assert text == "hello"
            assert kwargs["max_tokens"] == 12
            yield np.array([0.0, 0.5, -0.5], dtype=np.float32)

    def fake_load_model(self):
        self.model = FakeModel()
        self.sample_rate = FakeModel.sample_rate

    def fake_set_voice(self, voice):
        self.current_voice = PocketTTSVoice(str(voice))
        self.current_voice_state = "fake-state"

    monkeypatch.setattr(PocketTTSEngine, "_load_model", fake_load_model)
    monkeypatch.setattr(PocketTTSEngine, "set_voice", fake_set_voice)

    engine = PocketTTSEngine(voice="alba", max_tokens=12)
    try:
        assert engine.synthesize("hello") is True
        assert engine.audio_duration == 3 / 24000
        assert engine.queue.get_nowait() == np.array(
            [0, 16383, -16383], dtype=np.int16
        ).tobytes()
    finally:
        engine.shutdown()


def test_pocket_engine_shutdown_stops_worker(monkeypatch):
    monkeypatch.setattr(PocketTTSEngine, "_load_model", lambda self: None)
    monkeypatch.setattr(PocketTTSEngine, "set_voice", lambda self, voice: None)

    engine = PocketTTSEngine()
    worker = engine._synthesis_worker_thread

    engine.shutdown()

    assert not worker.is_alive()
    assert engine.model is None


def test_pocket_engine_loads_model_on_requested_device(monkeypatch):
    loaded_models = []

    class FakeModel:
        sample_rate = 24000

        def __init__(self):
            self.moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

    class FakeTTSModel:
        @classmethod
        def load_model(cls, **kwargs):
            model = FakeModel()
            loaded_models.append((kwargs, model))
            return model

    fake_pocket_tts = types.SimpleNamespace(TTSModel=FakeTTSModel)
    monkeypatch.setitem(sys.modules, "pocket_tts", fake_pocket_tts)
    monkeypatch.setattr(PocketTTSEngine, "set_voice", lambda self, voice: None)

    engine = PocketTTSEngine(
        voice="alba",
        model_config="custom.yaml",
        device="cuda",
    )
    try:
        kwargs, model = loaded_models[0]
        assert kwargs == {"config": "custom.yaml"}
        assert model.moved_to == "cuda"
        assert engine.sample_rate == 24000
    finally:
        engine.shutdown()
