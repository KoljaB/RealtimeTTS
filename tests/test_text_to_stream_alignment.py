import threading
import time

import numpy as np
import pytest
import pyaudio

from RealtimeTTS import BaseEngine, TextToAudioStream
from RealtimeTTS.alignment import FakeWordAligner, WordAlignment
from RealtimeTTS.alignment.capture import CapturedAudioSegment
import RealtimeTTS.stream_player as stream_player
import RealtimeTTS.text_to_stream as text_to_stream


class _FakeOutputStream:
    def write(self, chunk):
        time.sleep(0.001)

    def is_active(self):
        return True

    def start_stream(self):
        pass

    def stop_stream(self):
        pass

    def close(self):
        pass


class _FakePyAudio:
    def open(self, *args, **kwargs):
        return _FakeOutputStream()

    def get_sample_size(self, stream_format):
        if stream_format == pyaudio.paInt16:
            return 2
        if stream_format == pyaudio.paFloat32:
            return 4
        return 2

    def get_default_output_device_info(self):
        return {"index": 0}

    def get_device_info_by_index(self, index):
        return {
            "maxOutputChannels": 1,
            "defaultSampleRate": 16000,
            "hostApi": 0,
            "name": "fake",
            "maxInputChannels": 0,
        }

    def get_host_api_info_by_index(self, index):
        return {"name": "fake"}

    def get_device_count(self):
        return 1

    def is_format_supported(self, rate, **kwargs):
        return rate == 16000


class _PCMEngine(BaseEngine):
    def post_init(self):
        self.engine_name = "pcm-test"
        self.sample_rate = 16000

    def get_stream_info(self):
        return pyaudio.paInt16, 1, self.sample_rate

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)
        audio = np.zeros(self.sample_rate, dtype=np.int16)
        self.queue.put(audio.tobytes())
        return True

    def get_voices(self):
        return []

    def set_voice(self, voice):
        pass

    def set_voice_parameters(self, **voice_parameters):
        pass


def _single_sentence(text):
    return [text]


def _two_sentences(text):
    return [f"{part.strip()}." for part in text.split(".") if part.strip()]


@pytest.mark.parametrize("language_tags", [False, True])
@pytest.mark.parametrize("worker_active", [False, True])
def test_input_iterator_failure_releases_sentence_worker(
    monkeypatch, language_tags, worker_active
):
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)

    class SentenceSplitter:
        @staticmethod
        def generate_sentences(source, **kwargs):
            sentence = ""
            for char in source:
                sentence += char
                if char == ".":
                    yield sentence
                    sentence = ""
            if sentence:
                yield sentence

    monkeypatch.setattr(text_to_stream, "_get_stream2sentence", lambda: SentenceSplitter)

    class StoppableEngine(_PCMEngine):
        def post_init(self):
            super().post_init()
            self.started = threading.Event()
            self.language = "en"
            self.stop_calls = 0

        def get_language(self):
            return self.language

        def set_language(self, language):
            self.language = language

        def synthesize(self, text, sentence_count=0):
            self.started.set()
            if not self.stop_synthesis_event.wait(5):
                raise AssertionError("input failure did not stop synthesis")
            return False

        def stop(self):
            self.stop_calls += 1
            super().stop()

    engine = StoppableEngine()
    stream = TextToAudioStream(engine, language_tags=language_tags, tokenizer="rule-based")
    workers = []
    sentence_queues = []
    failures = []
    real_thread = threading.Thread
    real_queue = text_to_stream.queue.Queue

    def record_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        if getattr(kwargs.get("target"), "__name__", "") == "synthesize_worker":
            workers.append(thread)
        return thread

    class RecordingQueue(real_queue):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sentence_queues.append(self)

    monkeypatch.setattr(text_to_stream.threading, "Thread", record_thread)
    monkeypatch.setattr(text_to_stream.queue, "Queue", RecordingQueue)
    source_error = RuntimeError("upstream text failed")

    def source():
        yield "Inside." if worker_active else "unfinished"
        if worker_active:
            assert engine.started.wait(2)
        raise source_error

    def play():
        try:
            stream.feed(source()).play(buffer_threshold_seconds=0)
        except BaseException as exc:
            failures.append(exc)

    play_thread = real_thread(target=play, daemon=True)
    try:
        play_thread.start()
        play_thread.join(3)
        assert not play_thread.is_alive()
        assert stream.error_flag
        assert len(workers) == 1
        assert not workers[0].is_alive()
        assert engine.stop_calls == 1
        assert not stream.abort_events
        assert not stream.is_playing()
        assert not stream.play_lock.locked()
        # Retain the existing public error-reporting behavior in both modes.
        assert failures == ([source_error] if language_tags else [])
    finally:
        engine.stop()
        for sentence_queue in sentence_queues:
            sentence_queue.put(None)
        for worker in workers:
            worker.join(2)
        play_thread.join(2)
        stream.close()


def test_fragment_lookahead_is_forwarded_to_stream2sentence(monkeypatch):
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)
    calls = []

    class RecordingSentenceSplitter:
        @staticmethod
        def generate_sentences(source, **kwargs):
            calls.append(kwargs)
            text = "".join(source)
            if text:
                yield text

    monkeypatch.setattr(
        text_to_stream,
        "_get_stream2sentence",
        lambda: RecordingSentenceSplitter,
    )
    stream = TextToAudioStream(_PCMEngine())
    stream.feed("Freut mich, das zu hören.").play(fragment_lookahead_words=8)

    assert calls
    assert calls[-1]["fragment_lookahead_words"] == 8


def test_fake_aligner_timings_reach_on_word(monkeypatch):
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)

    spoken_words = []
    aligner = FakeWordAligner(
        [
            WordAlignment("hello", 0.0, 0.08),
            WordAlignment("world", 0.1, 0.18),
        ]
    )

    stream = TextToAudioStream(
        _PCMEngine(),
        on_word=spoken_words.append,
        align_words=True,
        word_aligner=aligner,
        alignment_blocking=True,
        playout_chunk_size=3200,
        tokenizer="rule-based",
    )

    stream.feed("hello world.").play(tokenize_sentences=_single_sentence)

    assert [timing.word for timing in spoken_words] == ["hello", "world"]
    assert len(aligner.calls) == 1
    assert aligner.calls[0][0] == "hello world."


def test_align_words_false_does_not_call_aligner(monkeypatch):
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)

    aligner = FakeWordAligner([WordAlignment("hello", 0.0, 0.1)])
    stream = TextToAudioStream(
        _PCMEngine(),
        align_words=False,
        word_aligner=aligner,
        alignment_blocking=True,
        tokenizer="rule-based",
    )

    stream.feed("hello world.").play(tokenize_sentences=_single_sentence)

    assert aligner.calls == []


def test_alignment_offsets_are_stream_relative(monkeypatch):
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)

    spoken_words = []
    aligner = FakeWordAligner([WordAlignment("word", 0.0, 0.08)])
    stream = TextToAudioStream(
        _PCMEngine(),
        on_word=spoken_words.append,
        align_words=True,
        word_aligner=aligner,
        alignment_blocking=True,
        playout_chunk_size=3200,
        tokenizer="rule-based",
    )

    stream.feed("First sentence. Second sentence.").play(
        tokenize_sentences=_two_sentences
    )

    assert len(aligner.calls) == 2
    assert [timing.word for timing in spoken_words] == ["word", "word"]
    assert spoken_words[0].start_time == 0.0
    assert spoken_words[1].start_time == 1.0


def test_late_nonblocking_alignment_is_not_reused_by_a_later_playback(monkeypatch):
    monkeypatch.setattr(stream_player.pyaudio, "PyAudio", _FakePyAudio)
    started = threading.Event()
    release = threading.Event()

    class SlowAligner:
        def align_words(self, text, audio, sample_rate):
            started.set()
            assert release.wait(timeout=1.0)
            return [WordAlignment("stale", 0.0, 0.1)]

    engine = _PCMEngine()
    stream = TextToAudioStream(
        engine,
        align_words=True,
        word_aligner=SlowAligner(),
        tokenizer="rule-based",
    )
    segment = CapturedAudioSegment(
        text="old playback",
        start_time=0.0,
        audio_bytes=np.zeros(160, dtype=np.int16).tobytes(),
        stream_format=pyaudio.paInt16,
        channels=1,
        sample_rate=16000,
    )

    stream._schedule_alignment(segment)
    assert started.wait(timeout=1.0)
    stream._alignment_generation += 1
    release.set()
    for worker in stream._alignment_threads:
        worker.join(timeout=1.0)

    assert engine.timings.empty()
