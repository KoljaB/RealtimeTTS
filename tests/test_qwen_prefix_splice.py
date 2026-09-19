"""Signal provenance and CTC path tests for the offline listening experiment."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("prefix_splice", Path(__file__).parents[1] / "tools/qwen_prefix_splice.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_repeated_phones_require_blank():
    scores = np.full((5, 3), -20.0)
    scores[np.arange(5), [0, 1, 0, 1, 0]] = 0
    spans = demo.ctc_spans(scores, [1, 1], 0)
    assert [(a, b) for a, b, _ in spans] == [(1, 2), (3, 4)]
    with pytest.raises(ValueError, match="too short"):
        demo.ctc_spans(scores[:2], [1, 1], 0)


def test_ctc_advances_directly_between_distinct_phones():
    scores = np.full((2, 3), -20.0)
    scores[0, 1] = scores[1, 2] = 0
    assert [(a, b) for a, b, _ in demo.ctc_spans(scores, [1, 2], 0)] == [(0, 1), (1, 2)]


def alignment(items):
    return {"words": [{"word": w, "start": s, "end": e} for w, s, e in items]}


def test_exact_retention_and_crossfade():
    a = np.linspace(-0.8, 0.8, 1000, dtype=np.float32)
    b = np.linspace(0.7, -0.7, 2000, dtype=np.float32)
    a[400:500] = .05
    b[600:700] = -.05
    aa = alignment([("heute", 0.1, 0.4), ("morgen", 0.5, 0.9)])
    bb = alignment([("heute", 0.2, 0.6), ("morgen", 0.7, 1.1), ("gehen", 1.2, 1.8)])
    c, left, right, cuts = demo.splice(a, b, 1000, aa, bb)
    np.testing.assert_array_equal(left, a[:455])
    np.testing.assert_array_equal(right, b[645:])
    np.testing.assert_array_equal(c[:445], a[:445])
    np.testing.assert_array_equal(c[455:], b[655:])
    ramp = np.linspace(0, 1, 10, dtype=np.float32)
    np.testing.assert_allclose(c[445:455], a[445:455]*(1-ramp)+b[645:655]*ramp)
    assert len(c) == 455 + 1355 - 10
    assert cuts["overlap_samples"] == 10
    plain, _, _, _ = demo.splice(a, b, 1000, aa, bb, mode="gap")
    assert len(plain) == len(c)  # centered overlap does not shorten the gap
    np.testing.assert_array_equal(plain[:450], a[:450])
    np.testing.assert_array_equal(plain[450:], b[650:])
    bb["words"][1]["start"] = 0.5
    with pytest.raises(ValueError, match="unordered"):
        demo.splice(a, b, 1000, aa, bb)


def test_rms_refinement_finds_plateau_without_assuming_silence():
    audio = np.full(1000, .4, dtype=np.float32)
    audio[442:452] = .1
    center, report = demo.boundary_center(audio, 1000, .4, .5, 10)
    assert 445 <= center <= 450
    assert report["overlap_rms"] > 0
    assert abs(report["correction_ms"]) <= 5
    with pytest.raises(ValueError, match="No room"):
        demo.boundary_center(audio, 1000, .4, .405, 10)


def test_deeper_distant_valley_cannot_move_boundary():
    audio = np.full(1000, .4, dtype=np.float32)
    audio[410:435] = 0  # tempting but outside the allowed correction window
    center, report = demo.boundary_center(audio, 1000, .4, .5, 10)
    assert center == 450
    assert report["correction_ms"] == 0
    assert report["overlap_rms"] == pytest.approx(.4)
    with pytest.raises(ValueError, match="0 and 5"):
        demo.boundary_center(audio, 1000, .4, .5, 10, search_ms=20)


def test_rms_disabled_preserves_exact_midpoint_despite_valley():
    audio = np.full(1000, .4, dtype=np.float32)
    audio[440:450] = .1
    center, report = demo.boundary_center(audio, 1000, .4, .5, 10, search_ms=0)
    assert center == 450
    assert report["correction_ms"] == 0
    assert report["search_start_seconds"] == report["search_end_seconds"] == .45


def test_unicode_normalization():
    assert demo.words("Heute Morgen genießen wir's.") == ["heute", "morgen", "genießen", "wir's"]


def test_pcm_roundtrip(tmp_path):
    path = tmp_path / "test.wav"
    signal = np.array([-1, -.5, 0, .5, 32767/32768], dtype=np.float32)
    demo.write_wav(path, signal, 24000)
    result, rate = demo.read_wav(path)
    assert rate == 24000
    np.testing.assert_array_equal(signal, result)


def test_player_keys_replace_and_stop_playback(monkeypatch, tmp_path):
    (tmp_path / "D.wav").write_bytes(b"fixture")
    keys = iter("a1B2c3d4 q")
    calls = []
    monkeypatch.setitem(sys.modules, "msvcrt", types.SimpleNamespace(getwch=lambda: next(keys)))
    monkeypatch.setitem(sys.modules, "winsound", types.SimpleNamespace(
        SND_FILENAME=1, SND_ASYNC=2, PlaySound=lambda *args: calls.append(args)))
    demo.listen(tmp_path)
    assert [Path(path).name for path, flags in calls[:6]] == ["A.wav", "A.wav", "B.wav", "B.wav", "C.wav", "C.wav"]
    assert all(flags == 3 for _, flags in calls[:6])
    assert [Path(path).name for path, _ in calls[6:8]] == ["D.wav", "D.wav"]
    assert calls[8:] == [(None, 0), (None, 0)]
