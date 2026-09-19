import threading
import time

import numpy as np
import pytest

from RealtimeTTS import PrefixSpliceEngine
from RealtimeTTS.prefix_splice import words


class Backend:
    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        # Every word has 200 ms; unique synthesis offset exposes wrong sources.
        return np.linspace(.01*len(self.calls), .4, len(words(text))*200, dtype=np.float32)


class Aligner:
    def align(self, audio, rate, text):
        return {"words": [{"word": w, "start": i*.2+.01, "end": i*.2+.15}
                          for i, w in enumerate(words(text))]}


def engine(backend=None, **kwargs):
    return PrefixSpliceEngine(backend or Backend(), Aligner(), sample_rate=1000,
                             boundary_search_ms=0, max_queue_seconds=100, **kwargs)


def pcm(e):
    chunks = []
    while not e.queue.empty():
        chunks.append(e.queue.get_nowait())
    return np.frombuffer(b"".join(chunks), dtype="<i2")


def test_two_handoffs_no_replayed_or_missing_words():
    b = Backend(); e = engine(b)
    assert e.synthesize(iter(["one two three four five six."]))
    assert b.calls == ["one two", "one two three four", "one two three four five six."]
    merges = [x for x in e.metrics if x["event"] == "merge"]
    assert [x["after_word"] for x in merges] == [1, 3]
    actual = pcm(e)
    assert len(actual) == 1200  # both overlaps preserve duration at equal centers
    assert actual[0] == round(.01*32768)
    # Everything after the second handoff comes from final render #3.
    final = np.linspace(.03, .4, 1200, dtype=np.float32)
    expected = np.rint(final[585:]*32768).astype('<i2')
    np.testing.assert_array_equal(actual[585:], expected)


def test_continuous_playback_holds_audio_during_followup_and_preserves_merges():
    entered = threading.Event(); release = threading.Event()
    class SlowFollowup(Backend):
        def __call__(self, text):
            if len(self.calls) == 1:
                entered.set()
                assert release.wait(2)
            return super().__call__(text)
    backend = SlowFollowup()
    buffered = engine(backend, continuous_playback=True)
    text = "one two three four five six."
    result = []
    worker = threading.Thread(target=lambda: result.append(buffered.synthesize(text)))
    worker.start()
    assert entered.wait(1)
    assert buffered.queue.empty()  # no word fragment can reach the device
    assert any(x["event"] == "first_buffer_commit" for x in buffered.metrics)
    release.set(); worker.join(3)
    assert result == [True]
    immediate = engine()
    assert immediate.synthesize(text)
    assert backend.calls == immediate.synthesizer.calls
    np.testing.assert_array_equal(pcm(buffered), pcm(immediate))
    assert not [x for x in buffered.metrics if x["event"] == "estimated_underrun"]
    assert [x["after_word"] for x in buffered.metrics if x["event"] == "merge"] == [1, 3]


def test_continuous_cancel_discards_private_prefix():
    entered = threading.Event(); release = threading.Event()
    class SlowFollowup(Backend):
        def __call__(self, text):
            if self.calls:
                entered.set(); release.wait(2)
            return super().__call__(text)
    buffered = engine(SlowFollowup(), continuous_playback=True)
    worker = threading.Thread(target=lambda: buffered.synthesize("one two three four five."))
    worker.start(); assert entered.wait(1)
    buffered.stop(); release.set(); worker.join(2)
    assert buffered.queue.empty()
    assert buffered._pending_audio == []


def test_only_two_complete_words_trigger_and_tail_is_held():
    e = engine(); entered = threading.Event(); release = threading.Event()
    def source():
        yield "one tw"
        entered.set()
        release.wait(2)
        yield "o "
        # Block input until first audio is actually queued.
        deadline = time.monotonic()+2
        while e.queue.empty() and time.monotonic() < deadline:
            time.sleep(.005)
        assert not e.queue.empty()
        assert e.queue.qsize() <= 9  # only <=170ms, no held crossfade or word two
    worker = threading.Thread(target=e.synthesize, args=(source(),))
    worker.start(); assert entered.wait(1)
    assert e.synthesizer.calls == []
    release.set(); worker.join(3)
    assert not worker.is_alive()
    assert e.synthesizer.calls == ["one two"]
    assert len(pcm(e)) == 400  # EOF flushes retained audio exactly once


def test_long_no_punctuation_is_bounded():
    e = engine(max_block_words=6)
    assert e.synthesize(" ".join(["word"]*17))
    assert max(len(words(t)) for t in e.synthesizer.calls) <= 6
    assert len(e.synthesizer.calls) == 9  # 3 fixed-cost stages per bounded block
    assert len(pcm(e)) == 17*200


def test_stop_during_inference_discards_late_result_and_prevents_unsafe_reuse():
    started = threading.Event(); finish = threading.Event()
    def slow(text):
        started.set(); finish.wait(2)
        return np.ones(400, dtype=np.float32)*.1
    e = engine(slow)
    result = []
    worker = threading.Thread(target=lambda: result.append(e.synthesize("one two ")))
    worker.start(); assert started.wait(1)
    e.stop(); worker.join(.5)
    assert not worker.is_alive() and result == [False]
    assert e.queue.empty()
    with pytest.raises(RuntimeError, match="not finished"):
        e.synthesize("one")
    finish.set()
    for t in e._threads:
        t.join(1)
    assert e.queue.empty()


def test_stop_during_blocked_input_returns_without_stale_audio():
    blocked = threading.Event(); release = threading.Event()
    def source():
        blocked.set(); release.wait(2); yield "one two "
    e = engine(); worker = threading.Thread(target=e.synthesize, args=(source(),))
    worker.start(); assert blocked.wait(1)
    e.stop(); worker.join(.5)
    assert not worker.is_alive()
    release.set()
    for t in e._threads:
        t.join(1)
    assert e.queue.empty()


def test_short_final_inputs_and_delimiter_split():
    for text in ("one", "one two.", "one two three."):
        e = engine(); assert e.synthesize(iter(text))
        assert len(pcm(e)) == 200*len(words(text))


def test_alignment_failure_never_replays_first_word():
    e = engine()
    align = e.aligner.align
    def fail_second(audio, rate, text):
        if len(words(text)) > 2:
            raise ValueError("alignment failed")
        return align(audio, rate, text)
    e.aligner.align = fail_second
    with pytest.raises(ValueError, match="alignment failed"):
        e.synthesize("one two three four five.")
    assert len(pcm(e)) == 400  # recover A tail once, never duplicate first word
    assert e.last_error is not None


def test_late_second_stage_holds_overlap_and_reports_underrun():
    b = Backend()
    def slow_second(text):
        if len(words(text)) == 4:
            time.sleep(.25)  # W1's committed portion is only 175 ms
        return b(text)
    e = engine(slow_second)
    assert e.synthesize("one two three four five.")
    assert len(pcm(e)) == 1000
    gaps = [x for x in e.metrics if x["event"] == "estimated_underrun"]
    assert gaps and gaps[0]["gap_seconds"] > .05
    assert [x["after_word"] for x in e.metrics if x["event"] == "merge"] == [1, 3]


def test_stop_during_alignment_then_reuse():
    entered = threading.Event(); release = threading.Event()
    e = engine()
    normal = e.aligner.align
    def slow(*args):
        entered.set(); release.wait(2); return normal(*args)
    e.aligner.align = slow
    worker = threading.Thread(target=e.synthesize, args=("one two ",))
    worker.start(); assert entered.wait(1)
    e.stop(); worker.join(.5)
    assert not worker.is_alive() and e.queue.empty()
    release.set()
    for t in e._threads:
        t.join(1)
    e.aligner.align = normal
    assert e.synthesize("three")
    assert len(pcm(e)) == 200


def test_http_cancel_interrupts_active_read(monkeypatch):
    from RealtimeTTS.engines import prefix_splice_engine as module
    entered = threading.Event(); released = threading.Event()
    class Socket:
        def shutdown(self, how):
            released.set()
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            entered.set()
            assert released.wait(1)
            return b"cancelled"
    class Connection:
        def __init__(self, *args, **kwargs): self.sock = None
        def connect(self): self.sock = Socket()
        def request(self, *args, **kwargs): self.connect()
        def getresponse(self): return Response()
        def close(self): pass
    monkeypatch.setattr(module.http.client, "HTTPConnection", Connection)
    backend = module.QwenHttpSynthesizer("http://localhost", "test")
    worker = threading.Thread(target=backend._request, args=("/test",))
    worker.start(); assert entered.wait(1)
    backend.cancel(); worker.join(.5)
    assert not worker.is_alive()


def test_text_to_audio_stream_async_stop_with_blocked_source(monkeypatch):
    from RealtimeTTS import TextToAudioStream
    import RealtimeTTS.stream_player as player
    class Audio:
        def get_default_output_device_info(self): return {"index": 0}
        def get_sample_size(self, *args): return 2
    monkeypatch.setattr(player.pyaudio, "PyAudio", Audio)
    entered = threading.Event(); release = threading.Event()
    def source():
        entered.set(); release.wait(2); yield "one two "
    e = engine()
    stream = TextToAudioStream(e, muted=True)
    stream.feed(source()).play_async()
    assert entered.wait(1)
    start = time.monotonic()
    stream.stop()
    assert time.monotonic()-start < .5
    release.set()
    for t in e._threads:
        t.join(1)
    assert e.queue.empty()
