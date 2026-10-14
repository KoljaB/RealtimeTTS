"""Regression checks for coverage, numeric projection, and stop evidence."""
import numpy as np
import pytest

from RealtimeTTS.alignment.backends import CPUAlignmentModel
from RealtimeTTS.alignment.base import WordAlignment
from RealtimeTTS.alignment.evidence import coverage_counts
from RealtimeTTS.alignment.onnx_ctc import OnnxCTCBackend
from tests.test_alignment_controller import FakeEngine, _controller, _finish_model


@pytest.mark.parametrize("text,words,expected", [
    ("hello world", ["hello"], (5, 10)),
    ("go go", ["go"], (2, 4)),
    ("hello world", ["world"], (5, 10)),
    ("Hello, world!", ["hello", "world"], (10, 10)),
    ("A B", ["AB"], (2, 2)),
])
def test_coverage_requires_all_spoken_symbols(text, words, expected):
    assert coverage_counts(text, words) == expected


def test_partial_export_is_finished_but_never_ready():
    class OnlyPrefix:
        def align_words(self, text, audio, rate):
            return [WordAlignment("hello", 0, .05, .95)]
    controller = _controller(OnlyPrefix(), mode="export")
    engine = FakeEngine()
    try:
        controller.reset()
        controller.append_source("hello world")
        result = _finish_model(controller, controller.bind(engine), engine, "hello world")
        assert result.final and result.status == "partial"
        assert result.coverage == .5 and not result.errors
        assert result.to_dict()["coverage"] == .5
        assert [word.text for word in result.words] == ["hello"]
    finally:
        controller.close()


@pytest.mark.parametrize("text", ["5 cats", "at 16:00.", "Cedar42 and 5 cables"])
def test_numeric_projection_works_at_both_timestamp_levels(text):
    from num2words import num2words
    backend = object.__new__(OnnxCTCBackend)
    backend.language, backend._number_to_words = "english", num2words
    normalized, _ = backend.prepare_text(text)
    backend.tokens = {c: i + 1 for i, c in enumerate(sorted(set(normalized)))}
    backend.blank_id = 0
    path = [v for c in normalized for v in (backend.tokens[c], 0)]
    probabilities = np.full((len(path), len(backend.tokens) + 1), .00001)
    probabilities[np.arange(len(path)), path] = .999
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    backend.emissions = lambda audio, rate: np.log(probabilities)
    model = CPUAlignmentModel(backend, "english")
    words = model.align(text, np.zeros(100), 16000, level="word")
    characters = model.align(text, np.zeros(100), 16000, level="character")
    assert words["words"] == characters["words"]
    assert words["words"] and words["characters"] == []
    assert coverage_counts(text, (w["word"] for w in words["words"]))[0] == coverage_counts(text, ())[1]


def test_production_stream_rejects_weak_target_despite_full_alignment():
    from RealtimeTTS.alignment.streaming import StreamingTextAligner

    class WeakEvidence:
        blank_id = 0
        frame_seconds = .02
        right_context_seconds = .04
        def encode(self, text):
            return text, [1]
        def emissions(self, audio, rate):
            return np.log(np.tile([.01, .20, .78, .01], (8, 1)))

    events = []
    aligner = StreamingTextAligner(backend=WeakEvidence())
    session = aligner.start("a", 50, callback=events.append)
    try:
        session.feed(np.zeros(8))
        session.finish()
        assert session.done.wait(2)
        assert session.complete  # Advisory coverage is distinct from stop evidence.
        assert not session.completion_verified and not session.stop_requested.is_set()
        assert session.finish_reason == "text_aligned_unverified"
        assert not any(e["type"] == "alignment_complete" for e in events)
        assert events[-1]["completion_verified"] is False
        assert events[-1]["mean_log_margin"] < 0
    finally:
        aligner.close()


def test_moving_endpoint_cannot_stop_stream_on_one_confident_observation():
    import time
    from RealtimeTTS.alignment.streaming import StreamingTextAligner
    from tests.test_streaming_alignment import AcousticFixture

    aligner = StreamingTextAligner(backend=AcousticFixture())
    session = aligner.start("a", 50)
    try:
        for count in (8, 16, 24):
            session.feed(np.full(8, .1))
            deadline = time.monotonic() + 2
            while session.analyzed_seconds < count / 50:
                assert time.monotonic() < deadline
                time.sleep(.001)
            assert session.complete
            assert not session.stop_requested.is_set()
        session.finish()
        assert session.done.wait(2)
        assert session.completion_verified
    finally:
        session.finish(cancelled=True)
        aligner.close()


def test_multilingual_surplus_does_not_hide_a_completed_transcript():
    from RealtimeTTS.alignment.onnx_parakeet import match_tokens
    characters, words, complete = match_tokens(
        "a b", [" a", " b", " b"], [0, .16, .32], [-.01] * 3)
    assert complete and [w.word for w in words] == ["a", "b"]
    assert words[-1].end_time == pytest.approx(.24)


def test_multilingual_mismatch_retains_later_words_without_claiming_completion():
    from RealtimeTTS.alignment.onnx_parakeet import match_tokens
    _, words, complete = match_tokens(
        "hello world", [" hallo", " world"], [0, .48], [-.01] * 2)
    assert not complete
    assert [w.word for w in words] == ["world"]
    assert words[0].start_time == .48


def test_multilingual_inserted_character_is_not_a_complete_word():
    from RealtimeTTS.alignment.onnx_parakeet import match_tokens
    _, words, complete = match_tokens("ab", [" acb"], [0], [-.01])
    assert not complete and words == []


def test_mms_alignment_uses_local_ctc_and_preserves_normalized_space_offsets():
    from RealtimeTTS.alignment.torchaudio_fa import _align_mms_emissions
    probabilities = np.full((6, 3), .001)
    probabilities[np.arange(6), [0, 1, 0, 0, 2, 0]] = .998
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    result = _align_mms_emissions(np.log(probabilities), "a b", [[1], [2]], .02)
    assert [(c.character, c.index) for c in result] == [("a", 0), ("b", 2)]
    assert result[0].start_time == .02
    assert result[1].end_time == .1
    assert all(c.score > .99 for c in result)


def test_vectorized_ctc_matches_exhaustive_small_paths():
    from itertools import product
    from RealtimeTTS.alignment.ctc import ctc_forced_align, log_softmax
    rng = np.random.default_rng(20261009)
    for targets in ([1, 2], [1, 1], [2]):
        for _ in range(5):
            log_probs = log_softmax(rng.normal(size=(6, 3)))
            valid = []
            for path in product(range(3), repeat=6):
                collapsed = [token for i, token in enumerate(path)
                             if token != 0 and (i == 0 or token != path[i - 1])]
                if collapsed == list(targets):
                    valid.append((sum(log_probs[i, token] for i, token in enumerate(path)), path))
            best = max(valid)[1]
            expected = []
            previous = None
            for i, token in enumerate(best):
                if token != 0:
                    if token != previous:
                        expected.append([i, i + 1])
                    else:
                        expected[-1][1] = i + 1
                previous = token
            result = ctc_forced_align(log_probs, targets)
            assert [(span.start_frame, span.end_frame) for span in result] == [
                tuple(span) for span in expected]


def test_cpu_cursor_bounds_reanalysis_and_keeps_every_word():
    from tests.test_streaming_alignment import AcousticFixture

    class MeasuredAcoustics(AcousticFixture):
        def __init__(self):
            self.inputs = []
        def emissions(self, audio, sample_rate):
            self.inputs.append(len(audio) / sample_rate)
            return super().emissions(audio, sample_rate)

    backend = MeasuredAcoustics()
    model = CPUAlignmentModel(backend)
    text = " ".join(["a"] * 24)
    # One word each second, with explicit acoustic separators.
    one = np.array([.1] + [0.] * 46 + [.3, 0., 0.])
    audio = np.tile(one, 24)
    cursor = model.create_cursor()
    for end in range(25, len(audio) + 1, 25):
        result = cursor.align(text, audio[:end], 50)
    assert [word["word"] for word in result["words"]] == ["a"] * 24
    assert np.allclose([word["start_time"] for word in result["words"]], np.arange(24))
    assert max(backend.inputs) < 5
    assert sum(backend.inputs) < sum(np.arange(.5, 24.01, .5)) / 2
    assert cursor.audio_start > 0


def test_cpu_cursor_processes_long_final_audio_in_bounded_windows():
    from tests.test_streaming_alignment import AcousticFixture
    cursor = CPUAlignmentModel(AcousticFixture()).create_cursor()
    text = " ".join(["a"] * 30)
    audio = np.tile(np.array([.1] + [0.] * 46 + [.3, 0., 0.]), 30)
    result = cursor.align(text, audio, 50)
    assert len(result["words"]) == 30
    assert result["words"][-1]["start_time"] == pytest.approx(29)


def test_cpu_cursor_reports_loss_of_progress_instead_of_unbounded_inference():
    from tests.test_streaming_alignment import AcousticFixture
    cursor = CPUAlignmentModel(AcousticFixture()).create_cursor()
    with pytest.raises(RuntimeError, match="12-second"):
        cursor.align("a b", np.zeros(50 * 13), 50)


def test_cpu_cursor_keeps_number_projection_when_window_advances():
    from num2words import num2words
    from tests.test_streaming_alignment import AcousticFixture
    # The existing numeric projection regression also runs through a cursor.
    backend = object.__new__(OnnxCTCBackend)
    backend.language, backend._number_to_words = "english", num2words
    normalized, _ = backend.prepare_text("5 cats")
    backend.tokens = {c: i + 1 for i, c in enumerate(sorted(set(normalized)))}
    backend.blank_id = 0
    path = [v for c in normalized for v in (backend.tokens[c], 0)]
    values = np.full((len(path), len(backend.tokens) + 1), -12.)
    values[np.arange(len(path)), path] = -.01
    backend.emissions = lambda audio, rate: values
    result = CPUAlignmentModel(backend, "english").create_cursor().align("5 cats", np.zeros(100), 50)
    assert [w["word"] for w in result["words"]] == ["5", "cats"]


def test_completion_allows_isolated_weak_letters_but_requires_supported_suffix():
    from RealtimeTTS.alignment.base import CharacterAlignment
    from RealtimeTTS.alignment.evidence import CompletionPolicy
    text = "dark suit in greasy wash water all year"
    characters = [CharacterAlignment(c, i * .02, (i + 1) * .02, i, .99)
                  for i, c in enumerate(text)]
    from dataclasses import replace
    for i in (2, text.index("in"), text.rfind("a")):
        characters[i] = replace(characters[i], score=.001)
    assert CompletionPolicy().supports(characters, 2.0, 1.0)
    suffix = text.rfind("year")
    weak_tail = [replace(c, score=.01) if c.index >= suffix else c for c in characters]
    assert not CompletionPolicy().supports(weak_tail, 2.0, -1.0)
    assert not CompletionPolicy().supports(characters, 2.0, -1.0)


def test_completion_accepts_reduced_short_internal_word_but_not_missing_content_or_tail():
    from dataclasses import replace
    from RealtimeTTS.alignment.base import CharacterAlignment
    from RealtimeTTS.alignment.evidence import CompletionPolicy
    text = "bring the beautiful flowers home"
    chars = [CharacterAlignment(c, i * .02, (i + 1) * .02, i, .99)
             for i, c in enumerate(text)]
    policy = CompletionPolicy()
    def weak_word(word):
        start = text.index(word)
        return [replace(c, score=.01) if start <= c.index < start + len(word) else c
                for c in chars]
    assert policy.supports(weak_word("the"), 3, 3)
    assert not policy.supports(weak_word("flowers"), 3, 3)
    assert not policy.supports(weak_word("home"), 3, 3)
    assert not policy.supports(weak_word("the"), 3, -1)
    strict = CompletionPolicy(word_support_minimum_length=3)
    assert not strict.supports(weak_word("the"), 3, 3)


@pytest.mark.parametrize("tokens,probability,expected", [
    ([0, 3], .99, True),
    ([3], .89, False),
    ([2, 3], .99, False),
    ([0, 0], .99, False),
    ([], .99, False),
])
def test_ctc_boundary_requires_delimiter_before_any_extra_letter(tokens, probability, expected):
    from RealtimeTTS.alignment.evidence import ctc_end_boundary
    values = np.full((len(tokens), 4), (1 - probability) / 3)
    if tokens:
        values[np.arange(len(tokens)), tokens] = probability
    assert ctc_end_boundary(np.log(values), .2 + np.arange(len(tokens)) * .02,
                            .2, 0, 3) == expected


def test_confident_ctc_word_boundary_still_needs_tail_context():
    import time
    from RealtimeTTS.alignment.streaming import StreamingTextAligner
    from tests.test_streaming_alignment import AcousticFixture
    backend = AcousticFixture()
    backend.separator_id = 3
    aligner = StreamingTextAligner(backend=backend)
    session = aligner.start("a", 50)
    try:
        session.feed(np.array([.1, .1, 0, .3, 0, 0, 0, 0]))
        deadline = time.monotonic() + 2
        while session.analyzed_seconds < .16:
            assert time.monotonic() < deadline
            time.sleep(.001)
        assert not session.stop_requested.is_set()
        session.feed(np.zeros(8))
        assert session.done.wait(2)
        assert session.completion_verified and session.stop_requested.is_set()
        assert session.completion_boundary_observed
        assert session.inference_calls == 2
    finally:
        session.finish(cancelled=True)
        aligner.close()


@pytest.mark.parametrize("delimiter", [False, True])
def test_ctc_word_prefix_cannot_stop_even_at_end_of_stream(delimiter):
    from RealtimeTTS.alignment.streaming import StreamingTextAligner
    from tests.test_streaming_alignment import AcousticFixture
    backend = AcousticFixture()
    backend.separator_id = 3
    aligner = StreamingTextAligner(backend=backend)
    session = aligner.start("a", 50)
    try:
        session.feed(np.array([.1, .1, 0, .3 if delimiter else 0, .2, .2] + [0] * 30))
        session.finish()
        assert session.done.wait(2)
        assert session.complete
        assert not session.completion_verified
        assert not session.stop_requested.is_set()
        assert session.finish_reason == "text_aligned_unverified"
    finally:
        aligner.close()


def test_weak_last_character_cannot_hide_inside_a_supported_word():
    from RealtimeTTS.alignment.base import CharacterAlignment
    from RealtimeTTS.alignment.evidence import CompletionPolicy
    chars = [CharacterAlignment(c, i * .02, (i + 1) * .02, i, .99)
             for i, c in enumerate("door")]
    from dataclasses import replace
    chars[-1] = replace(chars[-1], score=.001)
    assert not CompletionPolicy().supports(chars, 5, 5)


@pytest.mark.parametrize("kwargs", [
    {"word_support_minimum_length": 0},
    {"word_support_minimum_length": True},
    {"minimum_boundary_probability": float("nan")},
    {"minimum_boundary_probability": 1.1},
])
def test_completion_policy_validates_added_thresholds(kwargs):
    from RealtimeTTS.alignment.evidence import CompletionPolicy
    with pytest.raises(ValueError):
        CompletionPolicy(**kwargs)


def test_first_word_onset_revision_does_not_erase_uncommitted_coverage():
    from RealtimeTTS.alignment.onnx_parakeet import match_tokens
    from RealtimeTTS.alignment.windows import window_anchor
    text = "por favor"
    previous, _, complete = match_tokens(
        text, [" Por", " favor"], [.08, .32], [-.01, -.01])
    assert complete
    offset, start, anchor = window_anchor(text, previous, 24000, .08, 3.5)
    revised, words, complete = match_tokens(
        text[offset:], [" Por", " favor"], [0, .24], [-.01, -.01],
        offset_seconds=start / 24000, anchor_seconds=anchor)
    assert complete and [word.word for word in words] == ["por", "favor"]
    assert revised[0].start_time == 0


def test_window_can_advance_after_a_mismatch_without_fabricating_coverage():
    from RealtimeTTS.alignment.base import CharacterAlignment
    from RealtimeTTS.alignment.windows import window_anchor
    text = "wrong later"
    chars = [CharacterAlignment(c, 1 + i * .1, 1.05 + i * .1, i, .99)
             for i, c in enumerate(text) if i >= 6]
    offset, start, anchor = window_anchor(text, chars, 1000, .02, 10)
    assert offset == 6 and start > 0 and anchor == pytest.approx(1.6)


def test_mms_loader_never_uses_removed_alignment_api(monkeypatch):
    import sys
    from types import SimpleNamespace
    from RealtimeTTS.alignment.torchaudio_fa import TorchaudioMMSFACharacterAligner
    calls = []
    model = SimpleNamespace(eval=lambda: None)
    bundle = SimpleNamespace(
        get_model=lambda **kwargs: (calls.append(kwargs), model)[1],
        get_tokenizer=lambda: lambda words: [[1] for word in words],
    )
    torch = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace(pipelines=SimpleNamespace(MMS_FA=bundle)))
    loaded = TorchaudioMMSFACharacterAligner()._load()
    assert loaded[:3] == (torch, bundle, model)
    assert calls == [{"with_star": False}]
