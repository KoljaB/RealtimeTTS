import asyncio
import json
import queue
import threading
import time

import numpy as np
import pytest

from RealtimeTTS.alignment import StreamingTextAligner
from RealtimeTTS.alignment.streaming import CTCPrefixTracker
from RealtimeTTS.engines.qwen_engine import QwenVoice
from tests.test_qwen_engine import FakeBackend, _engine, _queued_pcm
from tests.test_qwen_server import FakeEngine, _server, _register
from RealtimeTTS.qwen_server import create_app
from fastapi.testclient import TestClient


class AcousticFixture:
    blank_id = 0
    frame_seconds = 0.02
    right_context_seconds = 0.04
    vocabulary = {"a": 1, "b": 2, " ": 3}

    def encode(self, text):
        return text, [self.vocabulary[c] for c in text]

    def emissions(self, audio, sample_rate):
        stride = round(sample_rate * self.frame_seconds)
        frames = len(audio) // stride
        tokens = np.rint(audio[:frames * stride].reshape(frames, stride).mean(1) * 10).astype(int)
        result = np.full((frames, 4), -12.0)
        result[np.arange(frames), tokens] = -0.01
        return result


def probabilities(tokens):
    values = np.full((len(tokens), 4), -12.0)
    values[np.arange(len(tokens)), tokens] = -0.01
    return values


def test_partial_prefix_does_not_fabricate_missing_word():
    tracker = CTCPrefixTracker("a b", [1, 3, 2], 0)
    values = probabilities([1, 0, 3] + [0] * 100)
    tracker.advance(values, np.arange(len(values)) * 0.02)
    characters, words, complete = tracker.result(0.02)
    assert not complete
    assert "".join(c.character for c in characters) == "a "
    assert [w.word for w in words] == ["a"]


def test_repeated_character_requires_separate_acoustic_evidence():
    tracker = CTCPrefixTracker("aa", [1, 1], 0)
    tracker.advance(probabilities([1, 1, 1, 0]), np.arange(4) * 0.02)
    assert not tracker.result(0.02)[2]
    tracker.advance(probabilities([1, 0]), np.arange(4, 6) * 0.02)
    assert tracker.result(0.02)[2]


@pytest.mark.parametrize("guard", [True, False])
def test_complete_prefix_emits_both_timestamp_levels_and_completion_signal(guard):
    events = []
    aligner = StreamingTextAligner(backend=AcousticFixture())
    session = aligner.start("a b", 50, callback=events.append, stop_on_completion=guard)
    session.feed(np.array([0.1, 0, 0.3, 0, 0.2] + [0] * 40))
    session.finish()
    assert session.done.wait(2)
    aligner.close()
    assert session.error is None
    assert session.complete
    assert session.stop_requested.is_set() == guard
    assert session.finish_reason == "verified_text_end"
    completion = next(e for e in events if e["type"] == "alignment_complete")
    assert completion["stop_requested"] == guard
    assert completion["trim_end_time"] == pytest.approx(completion["end_time"] + 0.25)
    assert [w.word for w in session.words] == ["a", "b"]
    assert [c.index for c in session.characters] == [0, 1, 2]
    assert session.characters[-1].end_time == pytest.approx(0.1)
    assert events[-1]["type"] == "alignment_done"
    assert events[0]["words"][1]["start_time"] == pytest.approx(0.08)


def test_known_text_alignment_completes_despite_spelling_difference():
    expected, recognized = "current word", "currant word"
    vocabulary = {c: i + 1 for i, c in enumerate(sorted(set(expected + recognized)))}
    values = np.full((2 * len(recognized), len(vocabulary) + 1), -12.0)
    values[1::2, 0] = -0.01
    for i, c in enumerate(recognized):
        values[2 * i, vocabulary[c]] = -0.01
        values[2 * i, vocabulary[expected[i]]] = -0.8 if c != expected[i] else -0.01

    class SpellingFixture(AcousticFixture):
        def encode(self, text):
            return text, [vocabulary[c] for c in text]
        def emissions(self, audio, sample_rate):
            return values

    events = []
    aligner = StreamingTextAligner(backend=SpellingFixture())
    session = aligner.start(expected, 50, callback=events.append)
    session.feed(np.zeros(len(values)))
    session.finish()
    assert session.done.wait(2)
    aligner.close()
    assert session.error is None
    assert "".join(c.character for c in session.characters) == expected
    assert [w["word"] for e in events for w in e.get("words", [])] == ["current", "word"]
    assert session.complete
    assert session.stop_requested.is_set()
    completion = next(e for e in events if e["type"] == "alignment_complete")
    assert completion["end_time"] == pytest.approx(session.characters[-1].end_time)
    assert completion["trim_end_time"] == pytest.approx(completion["end_time"] + .25)


def test_display_alignment_does_not_force_missing_suffix():
    from RealtimeTTS.alignment.streaming import display_ctc_characters
    values = probabilities([1] + [0] * 30)
    characters = display_ctc_characters(values, [1, 2], "ab",
                                       np.arange(len(values)) * 0.02, 0, 0.02)
    assert "".join(c.character for c in characters) == "a"


def test_close_finishes_idle_sessions():
    aligner = StreamingTextAligner(backend=AcousticFixture())
    sessions = [aligner.start("a", 50) for _ in range(3)]
    aligner.close()
    assert all(s.done.is_set() for s in sessions)


class LoopingNative(FakeBackend):
    tail_value = 0.001

    def speaker_names(self):
        return ["vivian"]

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        for i in range(500):
            if kwargs["cancel_event"].is_set():
                self.cancel_observed.set()
                return
            value = {0: 0.1, 2: 0.3, 4: 0.2}.get(i, 0.0)
            # Nonzero quiet tail lets the test verify the actual fade.
            if i > 4:
                value = self.tail_value
            yield np.full(480, value, dtype=np.float32), 24000
            time.sleep(0.002)
        raise AssertionError("The endless native decoder was not stopped")


def test_guard_stops_native_decode_preserves_pcm_and_fades_tail(tmp_path):
    backend = LoopingNative()
    aligner = StreamingTextAligner(backend=AcousticFixture())
    engine = _engine(tmp_path, backend,
                     model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                     voice=QwenVoice("vivian", speaker="vivian"), trim_silence=False,
                     startup_buffer_ms=0, streaming_aligner=aligner,
                     fragment_fade_out_after_ms=0)
    try:
        for _ in range(2):
            assert engine.synthesize("a b")
            assert engine.last_alignment.done.wait(2)
            pcm = _queued_pcm(engine)
            assert len(pcm) > 2400
            assert len(pcm) < 24000 * 2
            assert np.max(pcm) > 5000
            assert pcm[-1] == 0
            assert not engine.last_synthesis_profile["cancelled"]
            assert engine.last_synthesis_profile["finish_reason"] == "verified_text_end"
            assert engine.last_synthesis_profile["ending_fade_samples"] == 600
    finally:
        engine.shutdown()
        aligner.close()


def test_guard_stops_native_decode_after_verified_tail(tmp_path):
    backend = LoopingNative()
    backend.tail_value = 0.03
    aligner = StreamingTextAligner(backend=AcousticFixture())
    events = []
    engine = _engine(
        tmp_path,
        backend,
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        voice=QwenVoice("vivian", speaker="vivian"),
        trim_silence=False,
        startup_buffer_ms=0,
        streaming_aligner=aligner,
        on_alignment=events.append,
        fragment_fade_out_after_ms=0,
    )
    try:
        assert engine.synthesize("a b")
        assert engine.last_alignment.done.wait(2)
        assert engine.last_synthesis_profile["finish_reason"] == "verified_text_end"
        assert engine.last_synthesis_profile["ending_fade_samples"] == 600
        assert not engine.last_synthesis_profile["cancelled"]
        completion = next(e for e in events if e["type"] == "alignment_complete")
        assert completion["trim_end_time"] == pytest.approx(completion["end_time"] + 0.25)
        assert engine.last_synthesis_profile["audio_duration_s"] >= completion["trim_end_time"]
        assert len(_queued_pcm(engine)) < 24000 * 2
    finally:
        engine.shutdown()
        aligner.close()


def test_slow_aligner_never_blocks_first_pcm_and_late_results_survive_cleanup(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class Slow(AcousticFixture):
        def emissions(self, audio, sample_rate):
            entered.set()
            assert release.wait(3)
            return super().emissions(audio, sample_rate)

    class Finite(LoopingNative):
        def stream(self, **kwargs):
            yield np.repeat(np.array([0.1, 0, 0.3, 0, 0.2] + [0] * 40), 480), 24000

    events = []
    aligner = StreamingTextAligner(backend=Slow())
    engine = _engine(tmp_path, Finite(),
                     model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                     voice=QwenVoice("vivian", speaker="vivian"),
                     trim_silence=False, streaming_aligner=aligner, on_alignment=events.append)
    try:
        assert engine.synthesize("a b")
        assert entered.wait(2)
        assert not engine.queue.empty()
        assert not engine.last_alignment.done.is_set()
        release.set()
        assert engine.last_alignment.done.wait(2)
        assert [w["word"] for e in events for w in e.get("words", [])] == ["a", "b"]
        assert engine.timings.qsize() == 2
    finally:
        release.set()
        engine.shutdown()
        aligner.close()


def test_websocket_delivers_fragment_relative_timestamps_before_done(tmp_path):
    class AlignedEngine(FakeEngine):
        streaming_aligner = object()
        hallucination_guard = True
        on_alignment = None
        last_alignment = None

        def synthesize(self, text):
            result = super().synthesize(text)
            self.last_alignment = object()
            self.on_alignment({
                "type": "alignment", "words": [{"word": text, "start_time": 0, "end_time": 0.1}],
                "characters": [], "complete": True,
            })
            self.on_alignment({"type": "alignment_done", "complete": True})
            return result

    server = _server(tmp_path, AlignedEngine())
    with TestClient(create_app(server)) as client:
        _register(client)
        assert client.get("/v1/capabilities").json()["features"]["streaming_alignment"]
        with client.websocket_connect("/v1/audio/speech-stream") as websocket:
            websocket.send_json({"type": "config", "voice": "mira", "language": "English"})
            websocket.send_json({"type": "text", "text": "This sentence has timestamps."})
            websocket.send_json({"type": "end"})
            events, pcm, ordered = [], [], []
            while True:
                message = websocket.receive()
                if message.get("bytes") is not None:
                    pcm.append(message["bytes"])
                    ordered.append("pcm")
                    continue
                event = json.loads(message["text"])
                events.append(event)
                ordered.append(event["type"])
                assert event["type"] != "error", event
                if event["type"] == "done":
                    break
        assert pcm
        assert ordered.index("fragment_start") < ordered.index("pcm")
        assert next(e for e in events if e["type"] == "fragment_start")["text"] == "This sentence has timestamps."
        boundary = ordered.index("fragment_end")
        assert boundary > ordered.index("pcm")
        for position, chunk in zip((i for i, kind in enumerate(ordered) if kind == "pcm"), pcm):
            if position > boundary:
                assert not any(chunk)  # Server-added gap silence is outside the fragment.
        assert ordered.index("fragment_end") < ordered.index("done")
        timestamps = [e for e in events if e["type"] == "alignment"]
        assert timestamps and timestamps[0]["fragment_index"] == 1
        assert timestamps[0]["request_id"] == events[-1]["request_id"]
        assert any(e["type"] == "alignment_done" for e in events)


def test_unresolved_prefix_stops_as_stall_without_claiming_completion():
    events = []
    aligner = StreamingTextAligner(backend=AcousticFixture(), stall_seconds=0.2)
    session = aligner.start("a b", 50, callback=events.append)
    session.feed(np.zeros(40, dtype=np.float32))
    assert session.done.wait(2)
    aligner.close()
    assert session.finish_reason == "alignment_stalled"
    assert session.stop_requested.is_set()
    assert not session.complete
    stalled = next(e for e in events if e["type"] == "alignment_stalled")
    assert stalled["characters_confirmed"] == 0


def test_unknown_prefix_is_not_aborted_in_timestamp_only_mode():
    aligner = StreamingTextAligner(backend=AcousticFixture(), stall_seconds=0.2)
    session = aligner.start("a b", 50, stop_on_completion=False)
    session.feed(np.zeros(40, dtype=np.float32))
    session.finish()
    assert session.done.wait(2)
    aligner.close()
    assert not session.stop_requested.is_set()
    assert not session.complete


def test_overlapping_windows_keep_every_word():
    text = " ".join(["ab"] * 45)
    tokens = [AcousticFixture.vocabulary[c] for c in text]
    audio = np.array([v for token in tokens for v in (token / 10, 0)], dtype=np.float32)
    aligner = StreamingTextAligner(backend=AcousticFixture())
    session = aligner.start(text, 50, stop_on_completion=False)
    try:
        for end in range(40, len(audio) + 40, 40):
            session.feed(audio[end - 40:end])
            if end >= len(audio):
                session.finish()
            deadline = time.monotonic() + 2
            while session.analyzed_seconds < min(end, len(audio)) / 50:
                assert time.monotonic() < deadline
                time.sleep(0.001)
        session.finish()
        assert session.done.wait(2)
        assert session.error is None
        assert session.complete
        assert [w.word for w in session.words] == text.split()
        assert [c.index for c in session.characters] == list(range(len(text)))
    finally:
        session.finish(cancelled=True)
        aligner.close()


@pytest.mark.parametrize("ending", ["...", "…"])
def test_guard_preserves_exact_native_input_including_ellipsis(tmp_path, ending):
    backend = LoopingNative()
    # Check the native request without running the intentionally endless fixture.
    engine = _engine(tmp_path, backend,
                     model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                     voice=QwenVoice("vivian", speaker="vivian"))
    try:
        assert engine._stream_kwargs("Let me see" + ending, engine.current_voice)["text"] == "Let me see" + ending
        engine.hallucination_guard = False
        assert engine._stream_kwargs("Let me see" + ending, engine.current_voice)["text"] == "Let me see" + ending
    finally:
        engine.shutdown()


def test_alternative_character_with_ten_percent_probability_is_not_confirmation():
    tracker = CTCPrefixTracker("a", [1], 0)
    values = np.log(np.array([[0.01, 0.2, 0.78, 0.01]]))
    tracker.advance(values, [0.0])
    assert tracker.result(0.02) == ([], [], False)


def test_word_separator_is_not_required_to_be_spoken():
    tracker = CTCPrefixTracker("a b", [1, 3, 2], 0, separator_id=3)
    tracker.advance(probabilities([1, 0, 2, 0, 3]), np.arange(5) * 0.02)
    characters, words, complete = tracker.result(0.02)
    assert complete
    assert [w.word for w in words] == ["a", "b"]
    assert "".join(c.character for c in characters) == "a b"




def test_finish_publishes_measured_tail_without_repeating_inference():
    events = []
    class DelayedRightContext(AcousticFixture):
        right_context_seconds = 0.16
    aligner = StreamingTextAligner(backend=DelayedRightContext())
    session = aligner.start("a b", 50, callback=events.append, stop_on_completion=False)
    try:
        session.feed(np.array([0] * 7 + [0.1]))
        deadline = time.monotonic() + 2
        while session.analyzed_seconds < 0.16:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        calls = session.inference_calls
        assert session.characters and not any(e["type"] == "alignment" for e in events)
        session.finish()
        assert session.done.wait(2)
        assert "".join(c["character"] for e in events for c in e.get("characters", [])) == "a"
        assert session.inference_calls == calls
        assert not session.complete
        assert not session.stop_requested.is_set()
    finally:
        aligner.close()



@pytest.mark.parametrize("source, spoken", [
    ("at 16:00.", "at sixteen hundred"),
    ("Cedar42 and 5 cables", "cedar forty two and five cables"),
])
def test_english_numeral_alignment_restores_measured_source_words(source, spoken):
    from num2words import num2words
    from RealtimeTTS.alignment.onnx_ctc import OnnxCTCBackend
    backend = object.__new__(OnnxCTCBackend)
    backend.language, backend._number_to_words = "english", num2words
    normalized, projection = backend.prepare_text(source)
    assert normalized == spoken
    events = []
    callback = projection(events.append)
    # Publish incrementally, including a partial numeral. No fabricated digit span.
    for i, character in enumerate(normalized):
        callback({"type":"alignment", "complete":False, "words":[],
                  "characters":[{"character":character, "index":i,
                                 "start_time":i * .02, "end_time":(i+1) * .02, "score":.9}]})
    words = [w for e in events for w in e["words"]]
    assert [w["word"] for w in words] == source.lower().rstrip(".").split()
    numeral = next(w for w in words if any(c.isdigit() for c in w["word"]))
    expected = "sixteen hundred" if "16:00" in source else "cedar forty two"
    assert numeral["start_time"] == pytest.approx(normalized.index(expected) * .02)
    assert numeral["end_time"] == pytest.approx((normalized.index(expected) + len(expected)) * .02)
    characters = [c for e in events for c in e["characters"]]
    assert "".join(c["character"] for c in characters) == source.lower().rstrip(".").replace(" ", "")
    assert [c["character"] for c in characters if any(v.isdigit() for v in c["character"])] == (
        ["16:00"] if "16:00" in source else ["cedar42", "5"])


def test_finished_text_stops_immediately_without_quiet_tail():
    events = []

    class LateFinalWord(AcousticFixture):
        right_context_seconds = 0.12

    aligner = StreamingTextAligner(backend=LateFinalWord())
    session = aligner.start("a b", 50, callback=events.append)
    try:
        audio = np.array([0.1, 0, 0.3, 0, 0.2] + [0.03] * 35)
        for offset in range(0, audio.size, 8):
            session.feed(audio[offset:offset + 8])
            if session.done.wait(0.25):
                break
        assert session.done.is_set()
        assert session.complete
        assert session.stop_requested.is_set()
        assert session.finish_reason == "verified_text_end"
        completion = next(e for e in events if e["type"] == "alignment_complete")
        word_updates = [
            word["word"]
            for event in events
            if event["type"] == "alignment"
            for word in event["words"]
        ]
        assert word_updates == ["a", "b"]
        assert completion["finish_reason"] == "verified_text_end"
        assert completion["trim_end_time"] == pytest.approx(completion["end_time"] + 0.25)
        assert completion["analyzed_seconds"] - completion["end_time"] < 0.5
    finally:
        session.finish(cancelled=True)
        aligner.close()


def test_alignment_failure_and_long_pause_do_not_abort_by_default():
    aligner = StreamingTextAligner(backend=AcousticFixture())
    session = aligner.start("a b", 50)
    try:
        session.feed(np.array([0.1] + [0] * 399))
        deadline = time.monotonic() + 2
        while session.analyzed_seconds < 8:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert not session.complete
        assert not session.stop_requested.is_set()
    finally:
        session.finish(cancelled=True)
        aligner.close()


def test_language_specific_model_is_not_used_for_another_language():
    backend = AcousticFixture()
    backend.language = "english"
    aligner = StreamingTextAligner(backend=backend)
    try:
        with pytest.raises(ValueError, match="supports english"):
            aligner.start("a b", 50, language="german")
    finally:
        aligner.close()
