import pyaudio

from RealtimeTTS import BaseEngine, TextToAudioStream
import RealtimeTTS.stream_player as stream_player
import RealtimeTTS.text_to_stream as text_to_stream


class _FakePyAudio:
    pass


class _SentenceEngine(BaseEngine):
    def post_init(self):
        self.engine_name = "sentence-test"

    def get_stream_info(self):
        return pyaudio.paInt16, 1, 24_000

    def synthesize(self, text, sentence_count=0):
        return True

    def get_voices(self):
        return []

    def set_voice(self, voice):
        pass

    def set_voice_parameters(self, **voice_parameters):
        pass


class _PreloadSentenceEngine(_SentenceEngine):
    def post_init(self):
        super().post_init()
        self.preload_sentence_tokenizer = True


class _GeneratorEngine(_PreloadSentenceEngine):
    def post_init(self):
        super().post_init()
        self.can_consume_generators = True


def test_sentence_splitter_is_preloaded_for_sentence_engine(monkeypatch):
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)
    preload_calls = []
    monkeypatch.setattr(
        text_to_stream,
        "_get_stream2sentence",
        lambda: preload_calls.append(True),
    )

    TextToAudioStream(_PreloadSentenceEngine())

    assert preload_calls == [True]


def test_sentence_splitter_remains_lazy_by_default(monkeypatch):
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)
    preload_calls = []
    monkeypatch.setattr(
        text_to_stream,
        "_get_stream2sentence",
        lambda: preload_calls.append(True),
    )

    TextToAudioStream(_SentenceEngine())

    assert preload_calls == []


def test_sentence_splitter_is_not_preloaded_for_generator_engine(monkeypatch):
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)
    preload_calls = []
    monkeypatch.setattr(
        text_to_stream,
        "_get_stream2sentence",
        lambda: preload_calls.append(True),
    )

    TextToAudioStream(_GeneratorEngine())

    assert preload_calls == []
