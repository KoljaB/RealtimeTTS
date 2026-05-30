import numpy as np

from RealtimeTTS import PocketTTSGpuEngine, PocketTTSGpuVoice


def test_pocket_gpu_engine_streams_audio(monkeypatch):
    class FakeModel:
        sample_rate = 24000

        def generate_audio_stream(self, voice_state, text, **kwargs):
            assert voice_state == "fake-state"
            assert text == "hello"
            assert kwargs["teacher_forcing"] is True
            assert kwargs["frames_after_eos"] == 2
            yield np.array([0.0, 0.5, -0.5], dtype=np.float32)

    def fake_load_model(self):
        self.model = FakeModel()
        self.sample_rate = FakeModel.sample_rate

    def fake_set_voice(self, voice):
        self.current_voice = PocketTTSGpuVoice(str(voice))
        self.current_voice_state = "fake-state"

    monkeypatch.setattr(PocketTTSGpuEngine, "_load_model", fake_load_model)
    monkeypatch.setattr(PocketTTSGpuEngine, "set_voice", fake_set_voice)

    engine = PocketTTSGpuEngine(
        voice="alba",
        teacher_forcing=True,
        frames_after_eos=2,
    )
    try:
        assert engine.engine_name == "pocket_tts_gpu"
        assert engine.synthesize("hello") is True
        assert engine.audio_duration == 3 / 24000
        assert engine.queue.get_nowait() == np.array(
            [0, 16383, -16383], dtype=np.int16
        ).tobytes()
    finally:
        engine.shutdown()


def test_pocket_gpu_voice_representation():
    voice = PocketTTSGpuVoice("demo", audio_prompt_path="voice.wav")

    assert "voice.wav" in repr(voice)
