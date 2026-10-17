"""Bounded causal acoustic evidence for one synthesized PCM fragment.

This guard recognizes one narrow failure pattern: weak tonal audio in the
0.5-second window immediately after a quiet gap longer than one second. It is
an evidence gate, not a semantic hallucination detector. It never retries a
stream, treats silence as a failure, or retracts a prior start decision.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AcousticGuardDecision:
    """A one-shot causal decision for the current fragment."""

    reason: str
    onset_s: float
    evidence_s: float
    accepted_speech_s: float
    armed_at_s: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass
class _ResumeWindow:
    start_sample: int
    baseline_rms: float | None
    accepted_speech_s: float
    frames: list[np.ndarray]
    rms: list[float]
    quiet: list[bool]

    @property
    def samples(self) -> int:
        return sum(int(frame.size) for frame in self.frames)

    def add(self, frame: np.ndarray, rms: float, quiet: bool) -> None:
        self.frames.append(frame.copy())
        self.rms.append(float(rms))
        self.quiet.append(bool(quiet))


class StreamingAcousticGuard:
    """Bounded, causal detector for weak tonal audio after a long pause.

    feed accepts mono floating-point PCM and returns one
    AcousticGuardDecision only when the narrow acoustic pattern is confirmed
    and at least gate_s of earlier non-quiet, unflagged material was accepted.
    Quiet frames and an incomplete or suspect resume window are never counted
    as accepted speech. After a suspect window, a complete recovery window must
    also reach normal-level energy relative to the frozen pre-noise baseline;
    merely failing the narrow tonal test does not release that quarantine. The
    caller should create one instance per fragment and stop/cancel only after
    handling the returned decision.

    The active RMS p95 uses a bounded history (12 seconds by default), so this
    is a causal approximation of the offline detector's preceding-active-frame
    baseline. diagnostics exposes evidence state; it is not a quality guarantee
    and no result means only that this pattern was not recognized. Active
    unflagged material is an operational approximation, not a semantic
    transcript or a promise of zero false positives.
    """

    REASON = "weak_tonal_audio_after_long_pause"
    # A quarantined run may resume only after a complete window reaches at
    # least half of the frozen pre-noise active RMS.  Failing the narrow
    # tonal test is not enough evidence that the run recovered.
    MIN_RECOVERY_RELATIVE_RMS = 0.5

    def __init__(
        self,
        sample_rate: int = 24_000,
        *,
        frame_s: float = 0.02,
        silence_dbfs: float = -50.0,
        long_silence_s: float = 1.0,
        resume_window_s: float = 0.5,
        gate_s: float = 1.0,
        baseline_window_s: float = 12.0,
    ) -> None:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate < 8_000:
            raise ValueError("sample_rate must be an integer >= 8000")
        for name, value in (
            ("frame_s", frame_s),
            ("long_silence_s", long_silence_s),
            ("resume_window_s", resume_window_s),
            ("gate_s", gate_s),
            ("baseline_window_s", baseline_window_s),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(silence_dbfs):
            raise ValueError("silence_dbfs must be finite")

        self.sample_rate = sample_rate
        self.frame_s = float(frame_s)
        self.silence_dbfs = float(silence_dbfs)
        self.long_silence_s = float(long_silence_s)
        self.resume_window_s = float(resume_window_s)
        self.gate_s = float(gate_s)
        self.baseline_window_s = float(baseline_window_s)
        self.frame_samples = max(1, round(sample_rate * self.frame_s))
        self.resume_samples = max(1, round(sample_rate * self.resume_window_s))
        self.gate_samples = max(1, round(sample_rate * self.gate_s))
        self._silence_threshold = 10 ** (self.silence_dbfs / 20.0)
        history_frames = max(1, int(np.ceil(self.baseline_window_s / self.frame_s)))
        self._active_rms: deque[float] = deque(maxlen=history_frames)

        self._pending = np.empty(0, dtype=np.float32)
        self._processed_samples = 0
        self._quiet_frames = 0
        self._quiet_start_sample: int | None = None
        self._accepted_samples = 0
        self._armed_at_s: float | None = None
        self._candidate: _ResumeWindow | None = None
        self._decision: AcousticGuardDecision | None = None
        self._resume_windows = 0
        self._suspect_resume_windows = 0
        self._unresolved_resume_windows = 0
        self._underpowered_recovery_windows = 0
        self._long_silence_runs = 0

        # Leading-island diagnostics are intentionally isolated from handling.
        self._leading_stage = "first"
        self._leading_start_sample: int | None = None
        self._leading_frames: list[np.ndarray] = []
        self._leading_gap_frames = 0
        self._leading_following_start_sample: int | None = None
        self._leading_second_frames: list[np.ndarray] = []
        self._leading_suspected: dict[str, Any] | None = None
        self._leading_deferred: list[tuple[float, int]] = []
        self._quarantine_active_run = False
        self._quarantine_candidate: _ResumeWindow | None = None
        self._quarantine_baseline_rms: float | None = None
        self._quarantine_boundary_seen = False

    @property
    def decision(self) -> AcousticGuardDecision | None:
        """The one-shot decision, if one has already been emitted."""

        return self._decision

    @property
    def armed(self) -> bool:
        """Whether the earlier non-quiet material has reached gate_s."""

        return self._armed_at_s is not None

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return concise bounded state and evidence counters."""

        candidate_samples = self._candidate.samples if self._candidate is not None else 0
        quarantine_samples = (
            self._quarantine_candidate.samples
            if self._quarantine_candidate is not None else 0
        )
        return {
            "sample_rate": self.sample_rate,
            "processed_s": self._processed_samples / self.sample_rate,
            "accepted_speech_s": self._accepted_samples / self.sample_rate,
            "armed": self.armed,
            "armed_at_s": self._armed_at_s,
            "pending_samples": int(self._pending.size),
            "resume_buffer_samples": candidate_samples,
            "quarantine_buffer_samples": quarantine_samples,
            "quarantine_boundary_seen": self._quarantine_boundary_seen,
            "quarantine_baseline_rms": self._quarantine_baseline_rms,
            "active_rms_history_frames": len(self._active_rms),
            "long_silence_runs": self._long_silence_runs,
            "resume_windows": self._resume_windows,
            "suspect_resume_windows": self._suspect_resume_windows,
            "unresolved_resume_windows": self._unresolved_resume_windows,
            "underpowered_recovery_windows": self._underpowered_recovery_windows,
            "leading_suspected": self._leading_suspected,
            "leading_deferred_samples": len(self._leading_deferred) * self.frame_samples,
            "quarantined_active_run": self._quarantine_active_run,
            "decision": self._decision.as_dict() if self._decision is not None else None,
        }

    @staticmethod
    def _spectrum(samples: np.ndarray, rate: int) -> dict[str, float]:
        p = np.abs(np.fft.rfft((samples - samples.mean()) * np.hanning(len(samples)))) ** 2
        hz = np.fft.rfftfreq(len(samples), 1 / rate)
        p[0] = 0
        total = float(p.sum())
        if total <= 1e-20:
            return {"low_band_fraction": 0.0, "flatness": 1.0}
        band = p[(hz >= 20) & (hz <= 4_000)]
        flat = float(
            np.exp(np.mean(np.log(np.maximum(band, 1e-30))))
            / max(float(band.mean()), 1e-30)
        )
        return {
            "low_band_fraction": float(p[(hz >= 20) & (hz < 300)].sum() / total),
            "flatness": flat,
        }

    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2)))

    def _accept_frame(self, rms: float, sample_count: int, timeline_end: int) -> None:
        if sample_count <= 0:
            return
        self._active_rms.append(float(rms))
        self._accepted_samples += int(sample_count)
        if self._armed_at_s is None and self._accepted_samples >= self.gate_samples:
            self._armed_at_s = timeline_end / self.sample_rate

    def _defer_leading_frame(self, rms: float, start_sample: int) -> None:
        self._leading_deferred.append((float(rms), int(start_sample)))

    def _release_leading_deferred(self) -> None:
        for rms, start_sample in self._leading_deferred:
            self._accept_frame(rms, self.frame_samples, start_sample + self.frame_samples)
        self._leading_deferred.clear()

    def _discard_leading_deferred(self) -> None:
        self._leading_deferred.clear()

    def _observe_leading(
        self, frame: np.ndarray, rms: float, quiet: bool, start_sample: int
    ) -> None:
        """Record a possible leading island without ever authorizing handling."""

        if self._leading_stage == "done" or self._leading_suspected is not None:
            return
        max_start = round(0.20 * self.sample_rate)
        max_duration = round(0.08 * self.sample_rate)
        min_gap = round(0.20 * self.sample_rate)
        min_following = round(0.12 * self.sample_rate)

        if self._leading_stage == "first":
            if quiet:
                if self._leading_frames:
                    self._leading_stage = "gap"
                    self._leading_gap_frames = 1
                elif start_sample >= max_start:
                    self._leading_stage = "done"
                return
            if not self._leading_frames:
                if start_sample >= max_start:
                    self._leading_stage = "done"
                    return
                self._leading_start_sample = start_sample
                self._leading_frames.append(frame.copy())
                return
            if len(self._leading_frames) * self.frame_samples >= max_duration:
                self._leading_stage = "done"
                self._leading_frames.clear()
                self._release_leading_deferred()
                return
            self._leading_frames.append(frame.copy())
            return

        if self._leading_stage == "gap":
            if quiet:
                self._leading_gap_frames += 1
                return
            if self._leading_gap_frames * self.frame_samples < min_gap:
                self._leading_stage = "done"
                self._leading_frames.clear()
                self._release_leading_deferred()
                return
            self._leading_stage = "second"
            self._leading_following_start_sample = start_sample
            self._leading_second_frames = [frame.copy()]
        elif self._leading_stage == "second":
            if quiet:
                self._leading_stage = "done"
                self._leading_frames.clear()
                self._leading_second_frames.clear()
                self._release_leading_deferred()
                return
            self._leading_second_frames.append(frame.copy())

        following_samples = sum(int(item.size) for item in self._leading_second_frames)
        if following_samples < min_following:
            return
        lead = np.concatenate(self._leading_frames)
        following = np.concatenate(self._leading_second_frames)[:min_following]
        lead_rms = self._rms(lead)
        following_rms = self._rms(following)
        relative = lead_rms / max(following_rms, 1e-12)
        spectral = self._spectrum(lead, self.sample_rate)
        leading_suspect = (
            relative <= 0.05
            and spectral["low_band_fraction"] >= 0.65
            and spectral["flatness"] <= 0.02
        )
        if leading_suspect:
            lead_start = self._leading_start_sample or 0
            lead_end = lead_start + len(lead)
            self._leading_suspected = {
                "reason": "isolated_weak_tonal_audio_before_speech",
                "start_s": lead_start / self.sample_rate,
                "end_s": lead_end / self.sample_rate,
                "following_audio_start_s": (
                    (self._leading_following_start_sample if self._leading_following_start_sample is not None else start_sample)
                    / self.sample_rate
                ),
                "rms_relative_to_following_audio": relative,
            }
            # A diagnostic leading island must never contribute to the gate.
            self._discard_leading_deferred()
        else:
            self._release_leading_deferred()
        self._leading_stage = "done"
        self._leading_frames.clear()
        self._leading_second_frames.clear()

    def _start_candidate(self, frame: np.ndarray, rms: float, start_sample: int) -> None:
        baseline = float(np.percentile(np.asarray(self._active_rms), 95)) if self._active_rms else None
        self._candidate = _ResumeWindow(
            start_sample=start_sample,
            baseline_rms=baseline,
            accepted_speech_s=self._accepted_samples / self.sample_rate,
            frames=[],
            rms=[],
            quiet=[],
        )
        self._candidate.add(frame, rms, False)
        self._quiet_frames = 0
        self._quiet_start_sample = None
        self._long_silence_runs += 1

    def _resume_rms(self, candidate: _ResumeWindow) -> float | None:
        if candidate.samples < self.resume_samples:
            return None
        audio = np.concatenate(candidate.frames)[: self.resume_samples]
        return self._rms(audio)

    def _classify_resume_window(self, candidate: _ResumeWindow) -> bool | None:
        """Classify a complete window against its pre-window RMS baseline."""

        if candidate.baseline_rms is None:
            return None
        resumed_rms = self._resume_rms(candidate)
        if resumed_rms is None:
            return None
        relative = resumed_rms / max(candidate.baseline_rms, 1e-12)
        audio = np.concatenate(candidate.frames)[: self.resume_samples]
        spectral = self._spectrum(audio, self.sample_rate)
        return bool(
            relative <= 0.25
            and spectral["low_band_fraction"] >= 0.65
            and spectral["flatness"] <= 0.02
        )

    def _recovery_is_qualified(self, candidate: _ResumeWindow) -> bool:
        """Require normal-level energy before releasing a suspect run."""

        if candidate.baseline_rms is None:
            return False
        resumed_rms = self._resume_rms(candidate)
        if resumed_rms is None:
            return False
        relative = resumed_rms / max(candidate.baseline_rms, 1e-12)
        return relative >= self.MIN_RECOVERY_RELATIVE_RMS

    def _accept_resume_window(self, candidate: _ResumeWindow) -> None:
        remaining = self.resume_samples
        consumed = 0
        for rms, quiet in zip(candidate.rms, candidate.quiet):
            count = min(self.frame_samples, remaining)
            if count <= 0:
                break
            if not quiet:
                self._accept_frame(rms, count, candidate.start_sample + consumed + count)
            consumed += count
            remaining -= count

    def _trailing_quiet_frames(self, candidate: _ResumeWindow) -> int:
        trailing = 0
        remaining = self.resume_samples
        for quiet in reversed(candidate.quiet):
            count = min(self.frame_samples, remaining)
            if count <= 0:
                break
            if not quiet:
                break
            trailing += 1
            remaining -= count
        return trailing

    def _start_quarantine_candidate(
        self, frame: np.ndarray, rms: float, start_sample: int
    ) -> None:
        candidate = _ResumeWindow(
            start_sample=start_sample,
            baseline_rms=self._quarantine_baseline_rms,
            accepted_speech_s=self._accepted_samples / self.sample_rate,
            frames=[],
            rms=[],
            quiet=[],
        )
        candidate.add(frame, rms, False)
        self._quarantine_candidate = candidate

    def _finish_candidate(self) -> AcousticGuardDecision | None:
        candidate = self._candidate
        if candidate is None or candidate.samples < self.resume_samples:
            return None
        self._candidate = None
        self._resume_windows += 1
        evidence_sample = candidate.start_sample + self.resume_samples
        suspect = self._classify_resume_window(candidate)
        if suspect is True:
            self._suspect_resume_windows += 1
            if candidate.accepted_speech_s >= self.gate_s and self._armed_at_s is not None:
                self._decision = AcousticGuardDecision(
                    reason=self.REASON,
                    onset_s=candidate.start_sample / self.sample_rate,
                    evidence_s=evidence_sample / self.sample_rate,
                    accepted_speech_s=candidate.accepted_speech_s,
                    armed_at_s=self._armed_at_s,
                )
            else:
                # Freeze the pre-noise baseline.  The remainder of this run
                # cannot become gate material until a fresh complete window
                # proves that it is no longer the same weak tonal pattern.
                self._quarantine_active_run = True
                self._quarantine_baseline_rms = candidate.baseline_rms
                self._quarantine_boundary_seen = False
        elif suspect is None:
            self._unresolved_resume_windows += 1
        else:
            # A clean resume window becomes accepted only after it is complete.
            self._accept_resume_window(candidate)
        # Preserve a trailing quiet run for the next causal gap, but never
        # retroactively remove any accepted start material.
        trailing = self._trailing_quiet_frames(candidate)
        if trailing:
            self._quiet_frames = trailing
            self._quiet_start_sample = evidence_sample - trailing * self.frame_samples
        else:
            self._quiet_frames = 0
            self._quiet_start_sample = None
        if self._quarantine_active_run:
            self._quarantine_boundary_seen = bool(trailing)
        return self._decision

    def _finish_quarantine_candidate(self) -> AcousticGuardDecision | None:
        candidate = self._quarantine_candidate
        if candidate is None or candidate.samples < self.resume_samples:
            return None
        self._quarantine_candidate = None
        self._resume_windows += 1
        evidence_sample = candidate.start_sample + self.resume_samples
        suspect = self._classify_resume_window(candidate)
        trailing = self._trailing_quiet_frames(candidate)
        if suspect is True:
            self._suspect_resume_windows += 1
            self._quarantine_boundary_seen = bool(trailing)
        elif suspect is None:
            self._unresolved_resume_windows += 1
            self._quarantine_boundary_seen = bool(trailing)
        elif not self._recovery_is_qualified(candidate):
            # A complete window that is merely unlike the narrow tonal
            # signature is still unsafe: it may be another weak continuation.
            self._underpowered_recovery_windows += 1
            self._quarantine_boundary_seen = bool(trailing)
        else:
            # Release only after a complete, non-suspect, normal-level window.
            # All material before this window remains quarantined.
            self._accept_resume_window(candidate)
            self._quarantine_active_run = False
            self._quarantine_baseline_rms = None
            self._quarantine_boundary_seen = False
        if trailing:
            self._quiet_frames = trailing
            self._quiet_start_sample = evidence_sample - trailing * self.frame_samples
        else:
            self._quiet_frames = 0
            self._quiet_start_sample = None
        return self._decision

    def _process_frame(self, frame: np.ndarray) -> AcousticGuardDecision | None:
        start_sample = self._processed_samples
        self._processed_samples += self.frame_samples
        rms = self._rms(frame)
        quiet = rms < self._silence_threshold
        self._observe_leading(frame, rms, quiet, start_sample)

        if self._candidate is not None:
            self._candidate.add(frame, rms, quiet)
            if self._candidate.samples >= self.resume_samples:
                return self._finish_candidate()
            return None

        if self._quarantine_active_run:
            if self._quarantine_candidate is not None:
                self._quarantine_candidate.add(frame, rms, quiet)
                if self._quarantine_candidate.samples >= self.resume_samples:
                    return self._finish_quarantine_candidate()
                return None
            if quiet:
                if self._quiet_frames == 0:
                    self._quiet_start_sample = start_sample
                self._quiet_frames += 1
                self._quarantine_boundary_seen = True
                return None
            if not self._quarantine_boundary_seen:
                # Discard the rest of the flagged active run, including
                # pulses, until at least one quiet frame opens a fresh window.
                self._quiet_frames = 0
                self._quiet_start_sample = None
                return None
            self._start_quarantine_candidate(frame, rms, start_sample)
            if self._quarantine_candidate.samples >= self.resume_samples:
                return self._finish_quarantine_candidate()
            return None

        if quiet:
            if self._quiet_frames == 0:
                self._quiet_start_sample = start_sample
            self._quiet_frames += 1
            return None

        if self._quiet_frames * self.frame_samples > self.long_silence_s * self.sample_rate + 1e-9:
            self._start_candidate(frame, rms, start_sample)
            if self._candidate.samples >= self.resume_samples:
                return self._finish_candidate()
            return None

        self._quiet_frames = 0
        self._quiet_start_sample = None
        if self._leading_stage in {"first", "gap"} and self._leading_suspected is None:
            self._defer_leading_frame(rms, start_sample)
            return None
        self._accept_frame(rms, self.frame_samples, start_sample + self.frame_samples)
        return None

    def feed(self, audio: np.ndarray) -> AcousticGuardDecision | None:
        """Consume a mono PCM chunk and return a new decision, if any."""

        if self._decision is not None:
            return None
        data = np.asarray(audio, dtype=np.float32)
        if data.ndim == 2 and data.shape[1] == 1:
            data = data[:, 0]
        if data.ndim != 1:
            raise ValueError("audio must be mono 1-D PCM")
        if data.size == 0:
            return None
        if not np.isfinite(data).all():
            raise ValueError("audio must contain finite PCM")
        work = np.concatenate((self._pending, data)) if self._pending.size else data
        complete = (work.size // self.frame_samples) * self.frame_samples
        self._pending = work[complete:].copy()
        if complete == 0:
            return None
        frames = work[:complete].reshape(-1, self.frame_samples)
        for frame in frames:
            decision = self._process_frame(frame)
            if decision is not None:
                return decision
        return None


__all__ = ["AcousticGuardDecision", "StreamingAcousticGuard"]
