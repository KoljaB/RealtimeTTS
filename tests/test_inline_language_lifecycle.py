from __future__ import annotations

import threading

import pytest

import RealtimeTTS.text_to_stream as text_module
from RealtimeTTS.inline_markup import InlineMarkupParser
from tests.test_inline_language import ObservingEngine, stream_factory
from tests.test_text_to_stream_alignment import _PCMEngine


def _observed_language(engine):
    return [
        (text.strip(), language)
        for text, language, _voice, _speed in engine.observations
        if text.strip()
    ]


class _BlockingGeneratorEngine(ObservingEngine):
    def post_init(self):
        super().post_init()
        self.can_consume_generators = True
        self.started = threading.Event()
        self.release = threading.Event()

    def synthesize(self, text, sentence_count=0):
        if not isinstance(text, str):
            text = "".join(text)
        if text.strip() == "Inside":
            self.started.set()
            if not self.release.wait(2):
                raise AssertionError("generator synthesis was not released")
        return super().synthesize(text, sentence_count)


class _BlockingWorkerEngine(ObservingEngine):
    def post_init(self):
        super().post_init()
        self.can_consume_generators = False
        self.started = threading.Event()
        self.release = threading.Event()

    def synthesize(self, text, sentence_count=0):
        if not isinstance(text, str):
            text = "".join(text)
        if text.strip() == "Inside":
            self.started.set()
            if not self.release.wait(2):
                raise AssertionError("worker synthesis was not released")
        return super().synthesize(text, sentence_count)


def test_generator_late_feed_keeps_an_open_language_scope(stream_factory):
    engine = _BlockingGeneratorEngine()
    stream = stream_factory(engine, language="en", language_tags=True)
    try:
        stream.feed("<es>Inside").play_async()
        assert engine.started.wait(2)

        # The first generator has reached EOF while its synthesis is blocked.
        # These two items land in the next iterator and must retain the parser scope.
        stream.feed("Tail")
        stream.feed("</es>")
        engine.release.set()

        thread = stream.play_thread
        assert thread is not None
        thread.join(2)
        assert not thread.is_alive()
        assert _observed_language(engine) == [("Inside", "es"), ("Tail", "es")]
        assert engine.language == "en"
    finally:
        engine.release.set()
        thread = stream.play_thread
        if thread is not None and thread.is_alive():
            thread.join(2)
        stream.close()


def test_worker_late_feed_keeps_splitter_and_engine_language_in_scope(
    stream_factory, monkeypatch
):
    calls = []

    class RecordingSplitter:
        @staticmethod
        def generate_sentences(source, **options):
            value = "".join(source)
            if value:
                calls.append((value.strip(), options["language"]))
                yield value

    monkeypatch.setattr(text_module, "_get_stream2sentence", lambda: RecordingSplitter)
    engine = _BlockingWorkerEngine()
    stream = stream_factory(engine, language="en", language_tags=True)
    try:
        stream.feed("<es>Inside").play_async()
        assert engine.started.wait(2)

        stream.feed("Tail")
        stream.feed("</es>")
        engine.release.set()

        thread = stream.play_thread
        assert thread is not None
        thread.join(2)
        assert not thread.is_alive()
        assert calls == [("Inside", "es"), ("Tail", "es")]
        assert _observed_language(engine) == [("Inside", "es"), ("Tail", "es")]
        assert engine.language == "en"
    finally:
        engine.release.set()
        thread = stream.play_thread
        if thread is not None and thread.is_alive():
            thread.join(2)
        stream.close()


def test_language_parser_failure_releases_sentence_worker(stream_factory, monkeypatch):
    created_threads = []
    play_failures = []
    stream = stream_factory(ObservingEngine(), language="en", language_tags=True)
    real_thread = text_module.threading.Thread

    def recording_thread(*args, **kwargs):
        target = kwargs.get("target")
        if target is None and args:
            target = args[0]

        if (
            getattr(target, "__self__", None) is stream
            and getattr(target, "__name__", "") == "play"
        ):
            def guarded_target(*target_args, **target_kwargs):
                try:
                    return target(*target_args, **target_kwargs)
                except BaseException as exc:
                    play_failures.append(exc)

            if "target" in kwargs:
                kwargs = dict(kwargs)
                kwargs["target"] = guarded_target
            else:
                args = (guarded_target, *args[1:])

        thread = real_thread(*args, **kwargs)
        created_threads.append((thread, target))
        return thread

    monkeypatch.setattr(text_module.threading, "Thread", recording_thread)

    sentence_queues = []
    real_queue = text_module.queue.Queue

    class RecordingQueue(real_queue):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sentence_queues.append(self)

    monkeypatch.setattr(text_module.queue, "Queue", RecordingQueue)

    source = "<es>" * (InlineMarkupParser.MAX_LANGUAGE_DEPTH + 1)
    try:
        stream.feed(source).play_async()
        play_thread = stream.play_thread
        assert play_thread is not None
        play_thread.join(2)
        assert not play_thread.is_alive()
        assert [type(exc) for exc in play_failures] == [ValueError]
        assert stream.error_flag

        worker_threads = [
            thread
            for thread, target in created_threads
            if getattr(target, "__name__", "") == "synthesize_worker"
        ]
        assert all(not thread.is_alive() for thread in worker_threads)
    finally:
        # This releases a worker if the implementation fails the assertion above;
        # it keeps this regression test from leaving a daemon blocked in CI.
        for sentence_queue in sentence_queues:
            sentence_queue.put(None)
        for thread, target in created_threads:
            if getattr(target, "__name__", "") == "synthesize_worker":
                thread.join(2)
        stream.close()


class _FailingPrimaryEngine(ObservingEngine):
    def post_init(self):
        super().post_init()
        self.failed = threading.Event()

    def synthesize(self, text, sentence_count=0):
        self.failed.set()
        return False


class _UnsupportedFallbackEngine(_PCMEngine):
    def post_init(self):
        super().post_init()
        self.language = "en"


def test_unsupported_fallback_language_fails_visibly_without_hanging(stream_factory):
    primary = _FailingPrimaryEngine()
    fallback = _UnsupportedFallbackEngine()
    stream = stream_factory(
        [primary, fallback],
        language="en",
        language_tags=True,
    )
    try:
        with pytest.raises(NotImplementedError, match="inline language"):
            stream.feed("<es>Inside</es>").play()
        assert primary.failed.is_set()
        assert not stream.is_playing()
    finally:
        stream.close()