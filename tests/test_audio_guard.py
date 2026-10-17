"""Focused causal tests for the bounded streaming acoustic guard."""
from __future__ import annotations

import numpy as np
import pytest

from RealtimeTTS.audio_guard import AcousticGuardDecision, StreamingAcousticGuard

RATE = 24_000


def tone(seconds: float, amplitude: float, frequency: float) -> np.ndarray:
    count = round(seconds * RATE)
    return (amplitude * np.sin(2 * np.pi * frequency * np.arange(count) / RATE)).astype(np.float32)


def noise(seconds: float, amplitude: float, seed: int = 4) -> np.ndarray:
    count = round(seconds * RATE)
    return (amplitude * np.random.default_rng(seed).standard_normal(count)).astype(np.float32)


def bad_continuation(prefix_s: float = 1.2) -> np.ndarray:
    return np.concatenate(
        (
            tone(prefix_s, 0.25, 440),
            np.zeros(round(1.1 * RATE), np.float32),
            tone(0.5, 0.01, 150),
        )
    )


def feed_partitioned(guard: StreamingAcousticGuard, audio: np.ndarray, seed: int = 7):
    rng = np.random.default_rng(seed)
    offset = 0
    decision = None
    while offset < audio.size:
        size = int(rng.integers(1, 2401))
        decision = guard.feed(audio[offset : offset + size])
        offset += size
        if decision is not None:
            break
    return decision


def test_gate_requires_one_second_before_suspect_resume_window():
    guard = StreamingAcousticGuard(RATE)
    decision = guard.feed(bad_continuation())

    assert isinstance(decision, AcousticGuardDecision)
    assert decision.reason == "weak_tonal_audio_after_long_pause"
    assert decision.onset_s == pytest.approx(2.3, abs=0.02)
    assert decision.evidence_s == pytest.approx(2.8, abs=0.02)
    assert decision.accepted_speech_s == pytest.approx(1.2, abs=0.02)
    assert decision.armed_at_s == pytest.approx(1.0, abs=0.02)
    assert guard.armed
    assert guard.feed(tone(0.2, 0.01, 150)) is None
    assert guard.diagnostics["decision"]["reason"] == decision.reason


def test_short_prefix_bypasses_suspect_window_and_quarantines_same_run():
    guard = StreamingAcousticGuard(RATE)
    assert guard.feed(bad_continuation(0.8)) is None
    assert guard.diagnostics["suspect_resume_windows"] == 1
    assert guard.diagnostics["accepted_speech_s"] == pytest.approx(0.8, abs=0.02)
    assert guard.diagnostics["quarantined_active_run"]

    # More of the same resumed run cannot arm the guard after the first .5 s.
    assert guard.feed(tone(0.4, 0.25, 440)) is None
    assert guard.decision is None
    assert guard.diagnostics["accepted_speech_s"] == pytest.approx(0.8, abs=0.02)

    # A quiet boundary opens a fresh candidate, but it still needs a
    # complete .5 s window before any material is credited.
    assert guard.feed(np.zeros(round(0.02 * RATE), np.float32)) is None
    assert guard.feed(tone(0.4, 0.25, 440)) is None
    assert guard.diagnostics["accepted_speech_s"] == pytest.approx(0.8, abs=0.02)
    assert not guard.armed
    assert guard.diagnostics["quarantined_active_run"]
    assert guard.feed(tone(0.1, 0.25, 440)) is None
    assert guard.diagnostics["accepted_speech_s"] == pytest.approx(1.3, abs=0.02)
    assert guard.armed
    assert not guard.diagnostics["quarantined_active_run"]


def test_short_quiet_then_same_weak_run_never_arms():
    guard = StreamingAcousticGuard(RATE)
    audio = np.concatenate(
        (
            tone(0.6, 0.25, 440),
            np.zeros(round(1.1 * RATE), np.float32),
            tone(0.5, 0.01, 150),
            np.zeros(round(0.02 * RATE), np.float32),
            tone(1.0, 0.01, 150),
        )
    )
    assert guard.feed(audio) is None
    state = guard.diagnostics
    assert state["accepted_speech_s"] == pytest.approx(0.6, abs=0.02)
    assert not state["armed"]
    assert state["quarantined_active_run"]
    assert state["suspect_resume_windows"] >= 2


def test_underpowered_nonpattern_does_not_release_quarantine():
    guard = StreamingAcousticGuard(RATE)
    audio = np.concatenate(
        (
            tone(0.6, 0.25, 440),
            np.zeros(round(1.1 * RATE), np.float32),
            tone(0.5, 0.01, 150),
            np.zeros(round(0.02 * RATE), np.float32),
            noise(0.5, 0.02),
            np.zeros(round(0.02 * RATE), np.float32),
            noise(1.0, 0.02, seed=9),
        )
    )
    assert guard.feed(audio) is None
    state = guard.diagnostics
    assert state["accepted_speech_s"] == pytest.approx(0.6, abs=0.02)
    assert not state["armed"]
    assert state["quarantined_active_run"]
    assert state["underpowered_recovery_windows"] >= 2


def test_suspect_run_is_quarantined_until_a_quiet_boundary():
    guard = StreamingAcousticGuard(RATE)
    audio = np.concatenate(
        (
            tone(0.6, 0.25, 440),
            np.zeros(round(2.0 * RATE), np.float32),
            tone(2.0, 0.01, 150),
        )
    )
    assert guard.feed(audio) is None
    state = guard.diagnostics
    assert state["suspect_resume_windows"] == 1
    assert state["accepted_speech_s"] == pytest.approx(0.6, abs=0.02)
    assert not state["armed"]
    assert state["quarantined_active_run"]


def test_quarantine_releases_only_after_boundary_and_new_clean_material():
    guard = StreamingAcousticGuard(RATE)
    audio = np.concatenate(
        (
            tone(0.6, 0.25, 440),
            np.zeros(round(1.1 * RATE), np.float32),
            tone(1.0, 0.01, 150),
            np.zeros(round(0.1 * RATE), np.float32),
            tone(0.5, 0.25, 440),
        )
    )
    assert guard.feed(audio) is None
    state = guard.diagnostics
    assert state["suspect_resume_windows"] == 1
    assert state["accepted_speech_s"] == pytest.approx(1.1, abs=0.02)
    assert state["armed"]


def test_incomplete_resume_window_never_classifies():
    guard = StreamingAcousticGuard(RATE)
    prefix = tone(1.2, 0.25, 440)
    gap = np.zeros(round(1.1 * RATE), np.float32)
    weak = tone(0.48, 0.01, 150)
    assert guard.feed(np.concatenate((prefix, gap, weak))) is None
    assert guard.diagnostics["resume_windows"] == 0
    assert guard.diagnostics["resume_buffer_samples"] > 0

    decision = guard.feed(tone(0.02, 0.01, 150))
    assert decision is not None
    assert guard.diagnostics["resume_windows"] == 1


def test_silence_and_ordinary_resume_never_abort():
    guard = StreamingAcousticGuard(RATE)
    ordinary = np.concatenate(
        (tone(1.2, 0.25, 440), np.zeros(round(1.4 * RATE), np.float32), tone(0.5, 0.25, 440))
    )
    assert guard.feed(ordinary) is None
    assert guard.diagnostics["long_silence_runs"] == 1
    assert guard.diagnostics["suspect_resume_windows"] == 0
    assert guard.diagnostics["accepted_speech_s"] == pytest.approx(1.7, abs=0.02)

    clean_noise = np.concatenate(
        (tone(1.2, 0.25, 440), np.zeros(round(1.1 * RATE), np.float32), noise(0.5, 0.12))
    )
    clean_guard = StreamingAcousticGuard(RATE)
    assert clean_guard.feed(clean_noise) is None


def test_leading_diagnostic_never_triggers_handling_or_retracts_start():
    guard = StreamingAcousticGuard(RATE)
    early = np.concatenate(
        (tone(0.04, 0.005, 150), np.zeros(round(0.36 * RATE), np.float32), tone(0.2, 0.25, 440))
    )
    assert guard.feed(early) is None
    assert guard.diagnostics["leading_suspected"] is not None
    # The <=80 ms leading island was diagnostic-only and did not credit the gate.
    accepted_before = guard.diagnostics["accepted_speech_s"]
    assert accepted_before == pytest.approx(0.2, abs=0.02)
    assert guard.diagnostics["leading_deferred_samples"] == 0
    assert guard.feed(tone(0.9, 0.25, 440)) is None
    assert guard.diagnostics["accepted_speech_s"] > accepted_before
    assert guard.decision is None


def test_chunk_partitioning_produces_identical_decision_and_state():
    audio = bad_continuation()
    whole = StreamingAcousticGuard(RATE)
    partitioned = StreamingAcousticGuard(RATE)
    direct = whole.feed(audio)
    split = feed_partitioned(partitioned, audio)
    assert direct is not None and split is not None
    assert direct.as_dict() == split.as_dict()
    for key in ("accepted_speech_s", "armed_at_s", "resume_windows", "suspect_resume_windows"):
        assert partitioned.diagnostics[key] == whole.diagnostics[key]


def test_buffers_remain_bounded_for_long_fragment():
    guard = StreamingAcousticGuard(RATE)
    assert guard.feed(tone(20.0, 0.25, 440)) is None
    state = guard.diagnostics
    assert state["pending_samples"] < guard.frame_samples
    assert state["resume_buffer_samples"] == 0
    assert state["active_rms_history_frames"] <= 600


def test_quiet_input_is_safe_and_does_not_abort():
    guard = StreamingAcousticGuard(RATE)
    assert guard.feed(np.zeros(100, np.float32)) is None
    assert guard.decision is None


def test_invalid_or_non_mono_input_is_rejected():
    guard = StreamingAcousticGuard(RATE)
    with pytest.raises(ValueError, match="mono"):
        guard.feed(np.zeros((10, 2), np.float32))
    with pytest.raises(ValueError, match="finite"):
        guard.feed(np.array([np.nan], np.float32))


def test_armed_is_read_only_and_diagnostics_are_bounded():
    guard = StreamingAcousticGuard(RATE)
    guard.feed(tone(1.0, 0.25, 440))
    assert guard.armed
    assert guard.diagnostics["armed"] is True
    assert guard.diagnostics["pending_samples"] < guard.frame_samples


def test_leading_island_does_not_credit_gate_before_resolution():
    guard = StreamingAcousticGuard(RATE)
    audio = np.concatenate(
        (
            tone(0.04, 0.005, 150),
            np.zeros(round(0.36 * RATE), np.float32),
            tone(0.9, 0.25, 440),
        )
    )
    assert guard.feed(audio) is None
    assert guard.diagnostics["leading_suspected"] is not None
    assert guard.diagnostics["accepted_speech_s"] == pytest.approx(0.9, abs=0.02)
    assert not guard.armed
