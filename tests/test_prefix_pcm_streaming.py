import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest

from RealtimeTTS import PrefixSpliceEngine, QwenHttpSynthesizer
from RealtimeTTS.prefix_splice import ctc_spans, words


class PrefixAligner:
    def align(self, audio, rate, text):
        return {"words": [{"word": word, "start": i*.2+.01, "end": i*.2+.15}
                          for i, word in enumerate(words(text))]}

    def align_prefix(self, audio, rate, text, boundary_count):
        result = self.align(audio, rate, " ".join(words(text)[:boundary_count+1]))
        result["words"][-1]["end"] = boundary_count*.2+.05
        result["phones"] = [{"word_index": i, "start": w["start"], "end": w["end"], "score": .99}
                            for i, w in enumerate(result["words"])]
        return result


class StreamingBackend:
    def __init__(self):
        self.sources = []
        self.before_eof = None

    def stream(self, text):
        audio = np.linspace(.01*(len(self.sources)+1), .4, 200*len(words(text))+300, dtype=np.float32)
        self.sources.append(audio)
        for start in range(0, len(audio), 100):
            if start >= len(audio)-100 and self.before_eof:
                self.before_eof(len(self.sources))
            yield audio[start:start+100]


def make(backend, aligner=None):
    return PrefixSpliceEngine(backend, aligner or PrefixAligner(), sample_rate=1000,
                             boundary_search_ms=0, max_queue_seconds=100, stream_initial_prefix=True)


def output(engine):
    chunks = []
    while not engine.queue.empty():
        chunks.append(engine.queue.get_nowait())
    return np.frombuffer(b"".join(chunks), dtype="<i2")


def test_commits_before_eof_and_preserves_exact_pcm_at_both_handoffs():
    backend = StreamingBackend(); engine = make(backend)
    def before_eof(stage):
        deadline = time.monotonic()+2
        def committed():
            return any(e["event"] == ("first_queue_commit" if stage==1 else "merge")
                       and (stage==1 or e["after_word"] == (1 if stage==2 else 3)) for e in engine.metrics)
        while not committed() and time.monotonic() < deadline:
            time.sleep(.005)
        assert committed(), "No commit before the producer's EOF"
    backend.before_eof = before_eof
    assert engine.synthesize("one two three four five six.")
    a,b,c = backend.sources
    ramp = np.linspace(0,1,10,dtype=np.float32)
    expected = np.concatenate((a[:175], a[175:185]*(1-ramp)+b[175:185]*ramp,
                               b[185:575], b[575:585]*(1-ramp)+c[575:585]*ramp,c[585:]))
    np.testing.assert_array_equal(output(engine), np.rint(expected*32768).astype('<i2'))
    assert not any(e["event"]=="stream_boundary_fallback" for e in engine.metrics)


def test_unstable_anchor_waits_for_complete_audio():
    class Drifting(PrefixAligner):
        calls = 0
        def align_prefix(self, *args):
            result = super().align_prefix(*args)
            self.calls += 1
            result["words"][0]["end"] += .02*(self.calls%2)
            return result
    backend = StreamingBackend(); e=make(backend, Drifting())
    assert e.synthesize("one two ")
    assert any(x["event"]=="stream_boundary_fallback" for x in e.metrics)
    np.testing.assert_array_equal(output(e), np.rint(backend.sources[0]*32768).astype('<i2'))


def test_incomplete_stream_tail_is_not_replayed_after_early_commit():
    class Broken(StreamingBackend):
        def stream(self, text):
            for chunk in super().stream(text):
                yield chunk
            deadline = time.monotonic()+2
            while e.queue.empty() and time.monotonic()<deadline:
                time.sleep(.005)
            raise OSError("broken stream")
    backend=Broken(); e=make(backend)
    with pytest.raises(OSError, match="broken stream"):
        e.synthesize("one two ")
    assert len(output(e)) == 175  # reserve from incomplete A must remain private


def test_ctc_free_suffix_does_not_stretch_last_known_phone():
    logits = np.full((7,4), -20.)
    logits[np.arange(7),[0,1,0,2,0,3,3]]=0
    spans=ctc_spans(logits,[1,2],0,allow_trailing=True)
    assert [(x[0],x[1]) for x in spans]==[(1,2),(3,4)]


def mock_response(backend, chunks, metadata=None):
    headers={"Content-Type":"audio/pcm", "X-Audio-Sample-Rate":"24000",
             "X-Audio-Channels":"1", "X-Audio-Bits-Per-Sample":"16", "X-Audio-Encoding":"s16le"}
    headers.update(metadata or {})
    class Response:
        def getheader(self, key, default=None): return headers.get(key,default)
        def read1(self, size): return next(iterator,b"")
    iterator=iter(chunks)
    @contextmanager
    def response(*args): yield Response()
    backend._response=response


def test_pcm_byte_split_and_metadata_contract():
    b=QwenHttpSynthesizer("http://localhost","test")
    raw=np.array([12,-234,32767,-32768],dtype='<i2').tobytes()
    mock_response(b,[raw[:1],raw[1:5],raw[5:]])
    np.testing.assert_array_equal(np.concatenate(list(b.stream("hello")))*32768,[12,-234,32767,-32768])
    mock_response(b,[b"x"])
    with pytest.raises(ValueError,match="Truncated"):
        list(b.stream("hello"))
    mock_response(b,[raw],{"X-Audio-Channels":"2"})
    with pytest.raises(ValueError,match="metadata"):
        list(b.stream("hello"))


def test_low_confidence_is_not_an_early_commit():
    class Uncertain(PrefixAligner):
        def align_prefix(self, *args):
            result=super().align_prefix(*args)
            result["phones"][-1]["score"]=.01
            return result
    b=StreamingBackend(); e=make(b,Uncertain())
    assert e.synthesize("one two ")
    assert any(x["event"]=="stream_boundary_fallback" for x in e.metrics)


def test_stop_during_stream_read_unblocks_and_has_no_late_pcm():
    from RealtimeTTS.engines.prefix_splice_engine import _Cancelled
    b=QwenHttpSynthesizer("http://localhost","test")
    entered=threading.Event(); release=threading.Event()
    headers={"Content-Type":"audio/pcm", "X-Audio-Sample-Rate":"24000", "X-Audio-Channels":"1",
             "X-Audio-Bits-Per-Sample":"16", "X-Audio-Encoding":"s16le"}
    class Response:
        def getheader(self,key,default=None): return headers.get(key,default)
        def read1(self,size):
            entered.set(); release.wait(2); raise OSError("socket shutdown")
    @contextmanager
    def response(*args): yield Response()
    b._response=response
    b.cancel=lambda: release.set()
    e=make(b)
    result=[]
    thread=threading.Thread(target=lambda: result.append(e.synthesize("one two ")))
    thread.start(); assert entered.wait(1)
    e.stop(); thread.join(.5)
    assert not thread.is_alive() and result==[False]
    assert e.last_error is None and e.queue.empty()


def test_failure_after_handoff_never_replays_previous_reserve():
    class BrokenSecond(StreamingBackend):
        def stream(self,text):
            for chunk in super().stream(text):
                yield chunk
                if len(self.sources)==2:
                    if any(x['event']=='merge' for x in e.metrics):
                        raise OSError('after handoff')
                    time.sleep(.01)
    b=BrokenSecond(); e=make(b)
    with pytest.raises(OSError,match='after handoff'):
        e.synthesize('one two three four five.')
    assert any(x['event']=='merge' for x in e.metrics)
    assert not any(x['event']=='retained_tail_recovery' for x in e.metrics)
    assert len(output(e)) < len(b.sources[1])
