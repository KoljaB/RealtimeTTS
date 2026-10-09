"""Multilingual matching and the first-published-PCM scheduling contract."""
import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from RealtimeTTS.alignment.onnx_parakeet import OnnxParakeetBackend, match_tokens, normalize_text
from RealtimeTTS.alignment.streaming import StreamingTextAligner
from RealtimeTTS.qwen_server import SynthesisState, _STREAM_END, _send_stream_pcm
from tests.test_qwen_server import _server
from tests.test_streaming_alignment import AcousticFixture


@pytest.mark.parametrize("text,tokens", [
    ("Das habe ich verstanden.", [" Das", " habe", " ich", " ver", "standen", "."]),
    ("Je l'ai compris.", [" Je", " l", "'", "ai", " compris", "."]),
    ("Sim, tamb\u00e9m.", [" Sim", ",", " tamb\u00e9m", "."]),
    ("\u042f \u0442\u043e\u0436\u0435 \u044d\u0442\u043e \u043f\u043e\u043d\u044f\u043b.", [" \u042f", " \u0442\u043e\u0436\u0435", " \u044d\u0442\u043e", " \u043f\u043e\u043d\u044f\u043b", "."]),
])
def test_native_subwords_match_words_without_forcing_text(text, tokens):
    chars, words, complete = match_tokens(
        normalize_text(text), tokens, np.arange(len(tokens)) * .16, [-.01] * len(tokens))
    assert complete
    assert [w.word for w in words] == normalize_text(text).split()
    assert all(w.end_time >= w.start_time and w.score > .9 for w in words)
    assert words[0].start_time == 0
    assert words[0].end_time == pytest.approx(.08)


@pytest.mark.parametrize("tokens,logprobs", [
    ([" a"], [-.01]),                    # Missing final word.
    ([" a", " c"], [-.01, -.01]),        # Wrong word.
    ([" a", " b", " b"], [-.01] * 3),    # Repeated unwanted word.
    ([" a", " b"], [-.01, -8]),          # Uncertain completion.
])
def test_missing_wrong_extra_or_uncertain_speech_cannot_complete(tokens, logprobs):
    assert not match_tokens("a b", tokens, np.arange(len(tokens)) * .16, logprobs)[2]


class MultilingualFixture:
    languages = OnnxParakeetBackend.languages
    language = None
    blank_id = -1
    separator_id = 32
    frame_seconds = .08
    right_context_seconds = .16
    timestamp_levels = ("word",)

    def encode(self, text):
        return text, [ord(c) for c in text]

    def align(self, audio, sample_rate, text, **kwargs):
        return match_tokens(text, [" a", " b"], [.04, .12], [-.01, -.01], **kwargs)


def test_language_routing_keeps_english_and_reports_actual_timestamp_levels():
    english = AcousticFixture()
    english.language = "english"
    multilingual = MultilingualFixture()
    aligner = StreamingTextAligner(backend=english, multilingual_backend=multilingual)
    english_session = aligner.start("a b", 50, language="english")
    events = []
    german_session = aligner.start("a b", 50, language="german", callback=events.append)
    assert english_session.backend is english
    assert german_session.backend is multilingual
    assert aligner.language_timestamps["english"] == ["word", "character"]
    assert aligner.language_timestamps["german"] == ["word"]
    with pytest.raises(ValueError, match="does not support"):
        aligner.start("a", 50, language="japanese")
    english_session.finish(cancelled=True)
    german_session.feed(np.array([.1] * 10 + [0.] * 50))
    german_session.finish()
    assert german_session.done.wait(2)
    aligner.close()
    assert german_session.error is None and german_session.complete
    assert any(e.get("words") for e in events)
    assert not any(e.get("characters") for e in events)


def _state():
    return SynthesisState("pcm", request_id="first-pcm", output_queue_chunks=4,
                          deadline=time.monotonic() + 5)


def test_websocket_publishes_pcm_before_releasing_model_inference():
    state, trace = _state(), []
    class ObservedBackend(AcousticFixture):
        def emissions(self, audio, sample_rate):
            trace.append("inference")
            return super().emissions(audio, sample_rate)
    class Socket:
        async def send_bytes(self, chunk):
            assert chunk == b"pcm"
            assert not trace
            assert not state.first_pcm_published.is_set()
            trace.append("published")
    aligner = StreamingTextAligner(backend=ObservedBackend())
    session = aligner.start("a", 50, start_after=state.first_pcm_published)
    session.feed(np.array([.1] + [0.] * 30))
    session.finish()
    state.output.put(b"pcm")
    state.output.put(_STREAM_END)
    asyncio.run(_send_stream_pcm(
        Socket(), state, synthesis_started=time.monotonic(), first_fragment_timings=None,
        session_id="s", cancelled=asyncio.Event()))
    assert session.done.wait(2)
    aligner.close()
    assert trace == ["published", "inference"]


def test_http_releases_alignment_only_after_first_yield(tmp_path):
    server = _server(tmp_path)
    state = _state()
    state.output.put(b"pcm")
    state.output.put(_STREAM_END)
    state.done.set()
    stream = server.iter_pcm(state)
    assert next(stream) == b"pcm"
    assert not state.first_pcm_published.is_set()
    with pytest.raises(StopIteration):
        next(stream)
    assert state.first_pcm_published.is_set()


def test_cancel_before_first_pcm_never_runs_inference():
    backend = AcousticFixture()
    backend.emissions = lambda *_: pytest.fail("cancelled request started inference")
    ready, cancelled = threading.Event(), threading.Event()
    aligner = StreamingTextAligner(backend=backend)
    session = aligner.start("a", 50, start_after=ready, cancel_event=cancelled)
    session.feed(np.zeros(100))
    cancelled.set()
    assert session.done.wait(2)
    aligner.close()
    assert session.inference_calls == 0
    assert session._cancelled


@pytest.mark.parametrize("language,expected", [("en", "english"), ("DE", "multilingual"), ("pt-br", "multilingual"), ("ru", "multilingual")])
def test_language_code_aliases_select_the_same_backend(language, expected):
    english = AcousticFixture()
    english.language = "english"
    multilingual = MultilingualFixture()
    aligner = StreamingTextAligner(backend=english, multilingual_backend=multilingual)
    session = aligner.start("a", 50, language=language)
    assert session.backend is (english if expected == "english" else multilingual)
    aligner.close()
