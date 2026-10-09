import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from RealtimeTTS._audio_backend import pyaudio
from RealtimeTTS.alignment.base import CharacterAlignment, WordAlignment
from RealtimeTTS.alignment.controller import AlignmentController
from RealtimeTTS.alignment.timeline import AlignmentConfig


RATE = 1_000


class FakeEngine:
    def __init__(self, *, native=False, clock=0.0):
        self.queue = queue.Queue()
        self.raw_queue = self.queue
        self.timings = queue.Queue()
        self._stream_format = (pyaudio.paInt16, 1, RATE)
        self.provides_word_timings = native
        self.audio_duration = float(clock)
        self.last_alignment = None
        self.on_alignment = None

    def get_stream_info(self):
        return self._stream_format


def _pcm(seconds=0.1):
    return np.zeros(round(seconds * RATE), dtype=np.int16).tobytes()


class FixedWords:
    supports_partial = False

    def align_words(self, text, audio, sample_rate):
        return [
            WordAlignment("café", 0.0, 0.20),
            WordAlignment("café", 0.20, 0.40),
        ]


class FixedCharacters:
    supports_partial = False

    def align_characters(self, text, audio, sample_rate):
        return [
            CharacterAlignment(character, index * 0.05, index * 0.05 + 0.04, index)
            for index, character in enumerate(text)
            if character != " "
        ]


class BlockingWords:
    supports_partial = False

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.completed = threading.Event()
        self.calls = []

    def align_words(self, text, audio, sample_rate):
        self.calls.append(text)
        self.entered.set()
        if not self.release.wait(3):
            raise AssertionError("test aligner was not released")
        self.completed.set()
        word = text.split()[0] if text.split() else ""
        return [WordAlignment(word, 0.0, 0.1)] if word else []


class SlowPartialWords(BlockingWords):
    supports_partial = True


def _controller(aligner, **kwargs):
    config = AlignmentConfig(aligner=aligner, **kwargs)
    return AlignmentController(config)


def _finish_model(controller, binding, engine, text, *, audio=True):
    binding.begin(text)
    if audio:
        engine.queue.put(_pcm())
    binding.finish()
    controller.finish()
    return controller.wait(timeout=2)


def test_native_timing_uses_engine_clock_and_includes_inserted_silence():
    controller = _controller(None, source="auto")
    engine = FakeEngine(native=True, clock=10.0)
    try:
        controller.reset()
        controller.append_source("hello")
        binding = controller.bind(engine)

        binding.silence(0.25)
        binding.begin("hello")
        engine.timings.put(SimpleNamespace(
            word="hello", start_time=10.10, end_time=10.30,
        ))
        binding.finish()
        controller.finish()

        snapshot = controller.wait(timeout=2)
        assert snapshot.final
        assert snapshot.status == "ready"
        assert snapshot.duration == pytest.approx(0.25)
        assert len(snapshot.words) == 1
        span = snapshot.words[0]
        assert span.text == "hello"
        assert span.source == "native"
        assert span.start_time == pytest.approx(0.35)
        assert span.end_time == pytest.approx(0.55)
    finally:
        controller.close()


def test_model_word_snapshot_preserves_repeated_unicode_original_offsets():
    text = "😀 café 😀 café"
    controller = _controller(FixedWords())
    engine = FakeEngine()
    try:
        controller.reset()
        controller.append_source(text)
        binding = controller.bind(engine)
        snapshot = _finish_model(controller, binding, engine, text)

        assert snapshot.status == "ready"
        assert [(span.text, span.text_start, span.text_end) for span in snapshot.words] == [
            ("café", 2, 6),
            ("café", 9, 13),
        ]
        assert all(span.source == "model" for span in snapshot.words)
        assert all(span.state == "final" for span in snapshot.words)
    finally:
        controller.close()


def test_model_character_snapshot_maps_python_codepoint_offsets():
    text = "é α é"
    controller = _controller(FixedCharacters(), level="character")
    engine = FakeEngine()
    try:
        controller.reset()
        controller.append_source(text)
        binding = controller.bind(engine)
        snapshot = _finish_model(controller, binding, engine, text)

        assert [span.text for span in snapshot.characters] == ["é", "α", "é"]
        assert [(span.text_start, span.text_end) for span in snapshot.characters] == [
            (0, 1), (2, 3), (4, 5),
        ]
        assert [span.text for span in snapshot.words] == ["é", "α", "é"]
    finally:
        controller.close()


@pytest.mark.parametrize("action", ["reset", "cancel"])
def test_stale_worker_result_is_fenced_after_reset_or_cancel(action):
    aligner = BlockingWords()
    controller = _controller(aligner)
    engine = FakeEngine()
    try:
        controller.reset()
        controller.append_source("old")
        binding = controller.bind(engine)
        binding.begin("old")
        engine.queue.put(_pcm())
        binding.finish()
        controller.finish()
        assert aligner.entered.wait(2)

        if action == "reset":
            controller.reset()
            before = controller.snapshot()
            assert before.status == "idle"
            assert not before.final
        else:
            controller.cancel()
            before = controller.snapshot()
            assert before.status == "cancelled"
            assert before.final

        aligner.release.set()
        assert aligner.completed.wait(2)
        after = controller.snapshot()
        assert after.words == ()
        assert after.characters == ()
        assert after.text == ("" if action == "reset" else "old")
        assert after.status == before.status
        assert after.final == before.final
    finally:
        aligner.release.set()
        controller.close()


def test_audio_is_queued_without_waiting_for_model_and_audio_limit_is_explicit():
    aligner = SlowPartialWords()
    controller = _controller(aligner, update_interval=0.01, max_audio_seconds=0.05)
    engine = FakeEngine()
    try:
        controller.reset()
        controller.append_source("hello")
        binding = controller.bind(engine)
        binding.begin("hello")
        payload = _pcm(0.10)

        started = time.monotonic()
        engine.queue.put(payload)
        elapsed = time.monotonic() - started
        assert elapsed < 0.5
        assert engine.raw_queue.get(timeout=1) == payload
        assert any("audio limit" in error.lower() for error in controller.snapshot().errors)
        assert not aligner.entered.wait(0.2)

        binding.finish()
        controller.finish()
        snapshot = controller.wait(timeout=2)
        assert snapshot.final
        assert snapshot.status == "unavailable"
    finally:
        aligner.release.set()
        controller.close()


def test_pending_alignment_queue_drops_oldest_fragment_at_bound():
    aligner = BlockingWords()
    controller = _controller(aligner, max_pending=1)
    engine = FakeEngine()
    try:
        controller.reset()
        controller.append_source("one two three")
        binding = controller.bind(engine)

        binding.begin("one")
        engine.queue.put(_pcm())
        binding.finish()
        assert aligner.entered.wait(2)

        binding.begin("two")
        engine.queue.put(_pcm())
        binding.finish()
        binding.begin("three")
        engine.queue.put(_pcm())
        binding.finish()

        controller.finish()
        aligner.release.set()
        snapshot = controller.wait(timeout=2)

        assert snapshot.status == "unavailable"
        assert any("queue limit" in error.lower() for error in snapshot.errors)
        assert [span.text for span in snapshot.words] == ["one", "three"]
    finally:
        aligner.release.set()
        controller.close()
