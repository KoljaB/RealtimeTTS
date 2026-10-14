"""Offline TTS quality check: long pauses and weak tonal audio after a pause.

This deliberately has no playback/engine integration or cloud calls. Its
thresholds were developed on nine human-reviewed recordings from
2026-10-13/14; no_anomaly_found is not a guarantee of perceptually clean speech.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import wave
import numpy as np

@dataclass(frozen=True)
class Config:
    frame_s: float = 0.02
    silence_dbfs: float = -50.0
    long_silence_s: float = 1.0
    resume_window_s: float = 0.5
    max_resume_relative_rms: float = 0.25
    min_low_band_fraction: float = 0.65
    max_flatness: float = 0.02
    leading_max_start_s: float = 0.20
    leading_max_duration_s: float = 0.08
    leading_min_gap_s: float = 0.20
    leading_resume_s: float = 0.12
    leading_max_relative_rms: float = 0.05

def read_wav(path: Path) -> tuple[np.ndarray, int, dict]:
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Expected a RIFF WAV")
    if int.from_bytes(data[4:8], "little") + 8 != len(data):
        raise ValueError("WAV size does not match the RIFF declaration")
    with wave.open(str(path), "rb") as w:
        rate, channels, width, count = (w.getframerate(), w.getnchannels(),
                                        w.getsampwidth(), w.getnframes())
        if width != 2 or w.getcomptype() != "NONE":
            raise ValueError("Supported input is uncompressed PCM16 WAV")
        if rate < 8000 or not 1 <= channels <= 8 or count == 0:
            raise ValueError("Invalid or empty audio")
        raw = w.readframes(count)
    if len(raw) != count * channels * width:
        raise ValueError("Truncated PCM data")
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float64) / 32768.0
    return samples, rate, {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                          "sample_rate": rate, "channels": channels,
                          "duration_s": count / rate}

def _spectrum(samples: np.ndarray, rate: int) -> dict:
    # Pick a channel by energy, rather than cancelling antiphase stereo.
    channel = int(np.argmax(np.mean(samples * samples, axis=0)))
    x = samples[:, channel]
    p = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))) ** 2
    hz = np.fft.rfftfreq(len(x), 1 / rate)
    p[0] = 0
    total = float(p.sum())
    if total <= 1e-20:
        return {"low_band_fraction": 0.0, "flatness": 1.0, "centroid_hz": 0.0}
    band = p[(hz >= 20) & (hz <= 4000)]
    flat = float(np.exp(np.mean(np.log(np.maximum(band, 1e-30)))) / max(float(band.mean()), 1e-30))
    return {"low_band_fraction": float(p[(hz >= 20) & (hz < 300)].sum() / total),
            "flatness": flat, "centroid_hz": float((p * hz).sum() / total)}

def _leading_noise(samples, rate, rms, starts, ends, quiet, config, first_word_end):
    # A detached, very weak low-tone island before sustained audio. A short
    # consonant or a pause alone is not enough; all gates must hold.
    changes = np.diff(np.r_[False, ~quiet, False].astype(np.int8))
    runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
    if len(runs) < 2:
        return None
    first, after = runs[0]
    following, following_after = runs[1]
    start, end = float(starts[first]), float(ends[after - 1])
    resume = float(starts[following])
    gap = resume - end
    if (start > config.leading_max_start_s + 1e-9
            or end - start > config.leading_max_duration_s + 1e-9
            or gap + 1e-9 < config.leading_min_gap_s
            or float(ends[following_after - 1]) - resume + 1e-9 < config.leading_resume_s):
        return None
    # A recognized quiet first word followed by another word is legitimate.
    # First-word START is unreliable: Whisper often assigns it to zero.
    if first_word_end is not None and first_word_end <= resume + config.frame_s:
        return None
    begin, finish = round(resume * rate), round((resume + config.leading_resume_s) * rate)
    if finish > len(samples):
        return None
    lead = samples[round(start * rate):round(end * rate)]
    following_audio = samples[begin:finish]
    lead_rms = float(np.sqrt(np.max(np.mean(lead * lead, axis=0))))
    following_rms = float(np.sqrt(np.max(np.mean(following_audio * following_audio, axis=0))))
    relative = lead_rms / max(following_rms, 1e-12)
    if relative > config.leading_max_relative_rms:
        return None
    spectral = _spectrum(lead, rate)
    if (spectral["low_band_fraction"] < config.min_low_band_fraction
            or spectral["flatness"] > config.max_flatness):
        return None
    acoustic_ready = finish / rate
    return {"kind": "suspected_strange_noise",
        "reason": "isolated_weak_tonal_audio_before_speech",
        "start_s": start, "end_s": end, "audible_leading_s": end - start,
        "following_quiet_gap_s": gap, "following_audio_start_s": resume,
        "leading_rms_dbfs": 20 * math.log10(max(lead_rms, 1e-12)),
        "following_rms_dbfs": 20 * math.log10(max(following_rms, 1e-12)),
        "rms_relative_to_following_audio": relative, **spectral,
        "first_asr_word_end_s": first_word_end,
        "acoustic_evidence_available_s": acoustic_ready,
        "evidence_available_s": len(samples) / rate if first_word_end is not None else acoustic_ready,
        "asr_guard_requires_full_result": first_word_end is not None}

def analyze(samples: np.ndarray, rate: int, config: Config = Config(), asr: dict | None = None) -> dict:
    started = time.perf_counter()
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.size == 0 or not np.isfinite(x).all() or rate < 8000:
        raise ValueError("Finite nonempty PCM audio at >=8 kHz is required")
    step = max(1, round(rate * config.frame_s))
    count = len(x) // step
    if count:
        frames = x[:count * step].reshape(count, step, x.shape[1])
        rms = np.sqrt(np.max(np.mean(frames * frames, axis=1), axis=1))
    else:
        rms = np.empty(0)
    if count * step < len(x):
        rms = np.r_[rms, np.sqrt(np.max(np.mean(x[count * step:] ** 2, axis=0)))]
    ends = np.minimum((np.arange(len(rms)) + 1) * step, len(x)) / rate
    starts = np.arange(len(rms)) * step / rate
    threshold = 10 ** (config.silence_dbfs / 20)
    quiet = rms < threshold
    transitions = np.diff(np.r_[False, quiet, False].astype(np.int8))
    gaps = []
    events = []
    for first, after in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
        start, end = float(starts[first]), float(ends[after - 1])
        duration = end - start
        if duration <= config.long_silence_s + 1e-9:
            continue
        first_detect = start + (math.floor(config.long_silence_s * rate / step) + 1) * step / rate
        gap = {"start_s": start, "end_s": end, "duration_s": duration,
               "evidence_available_s": min(end, first_detect)}
        gaps.append(gap)
        events.append({"kind": "long_silence", **gap, "threshold_dbfs": config.silence_dbfs})
        if after >= len(rms):
            continue
        begin = after * step
        finish = begin + round(config.resume_window_s * rate)
        if finish > len(x):
            continue  # Never pad an incomplete resume window into a clean/noise verdict.
        prior = rms[:first]
        active_prior = prior[prior >= threshold]
        if not len(active_prior):
            continue
        baseline = float(np.percentile(active_prior, 95))
        resumed = x[begin:finish]
        resumed_rms = float(np.sqrt(np.max(np.mean(resumed * resumed, axis=0))))
        relative = resumed_rms / max(baseline, 1e-12)
        spectral = _spectrum(resumed, rate)
        if (relative <= config.max_resume_relative_rms
                and spectral["low_band_fraction"] >= config.min_low_band_fraction
                and spectral["flatness"] <= config.max_flatness):
            events.append({"kind": "suspected_strange_noise", "reason": "weak_tonal_audio_after_long_pause",
                "start_s": begin / rate, "end_s": finish / rate,
                "evidence_available_s": finish / rate,
                "preceding_gap_s": duration, "rms_relative_to_prior_speech": relative,
                "rms_dbfs": 20 * math.log10(max(resumed_rms, 1e-12)), **spectral})
    first_word_end = None
    asr_status = "not_supplied"
    asr_evidence = None
    if asr is not None:
        if asr.get("error"):
            raise ValueError("ASR result contains an error")
        response = asr.get("transcription", asr.get("response", asr))
        words = response.get("words") or [word for seg in response.get("segments", [])
                                           for word in seg.get("words", [])]
        if not words:
            asr_status = "word_timestamps_unavailable"
        else:
            last = 0.0
            for word in words:
                a, b = float(word["start"]), float(word["end"])
                if not (math.isfinite(a) and math.isfinite(b) and 0 <= a <= b <= len(x) / rate + .02
                        and b >= last):
                    raise ValueError("Invalid ASR word timestamp")
                if first_word_end is None:
                    first_word_end = b
                last = b
            asr_status = "word_timestamps_available"
            asr_evidence = {"text": response.get("text"), "last_word_end_s": last,
                            "model": asr.get("model", response.get("model")),
                            "params": asr.get("params"), "timing": asr.get("timing")}
            tail = (starts >= last + .25) & ~quiet
            audible_s = float(np.sum(ends[tail] - starts[tail]))
            if len(x) / rate - last > 1 and audible_s >= .20:
                bounds = np.diff(np.r_[False, tail, False].astype(np.int8))
                intervals = [[float(starts[a]), float(ends[b - 1])]
                             for a, b in zip(np.flatnonzero(bounds == 1), np.flatnonzero(bounds == -1))]
                events.append({"kind": "suspected_strange_noise", "reason": "audible_audio_after_last_asr_word",
                    "start_s": intervals[0][0], "end_s": intervals[-1][1],
                    "audible_intervals_s": intervals, "last_asr_word_end_s": last,
                    "audible_tail_s": audible_s, "evidence_available_s": len(x) / rate})
    leading = _leading_noise(x, rate, rms, starts, ends, quiet, config, first_word_end)
    if leading is not None:
        events.insert(0, leading)
    noise = any(e["kind"] == "suspected_strange_noise" for e in events)
    return {"status": "noise_suspected" if noise else ("silence_suspected" if gaps else "no_anomaly_found"),
            "strange_noise_suspected": noise, "long_silence_suspected": bool(gaps),
            "events": events, "asr_status": asr_status, "asr_evidence": asr_evidence,
            "analysis_ms": (time.perf_counter() - started) * 1000,
            "config": asdict(config), "mode": "offline_full_audio",
            "qualification": "Developed on nine human-reviewed examples; no_anomaly_found is not proof of clean synthesis."}

def check_wav(path: Path, asr: dict | None = None) -> dict:
    x, rate, meta = read_wav(path)
    if asr is not None and (asr.get("sha256") or asr.get("source", {}).get("sha256")) != meta["sha256"]:
        raise ValueError("ASR result must be bound to the same WAV SHA256")
    return {"audio": meta, **analyze(x, rate, asr=asr)}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", type=Path)
    group.add_argument("--manifest", type=Path)
    parser.add_argument("--asr-json", type=Path, help="Hash-linked ASR result for --wav only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new result path")
    if args.manifest and args.asr_json:
        parser.error("--asr-json requires --wav")
    if args.wav:
        asr = json.loads(args.asr_json.read_text(encoding="utf-8-sig")) if args.asr_json else None
        result = check_wav(args.wav, asr)
    else:
        raw = args.manifest.read_bytes()
        corpus = json.loads(raw.decode("utf-8-sig"))
        records = []
        for rec in corpus["records"]:
            checked = check_wav(Path(rec["path"]))
            if checked["audio"]["sha256"] != rec["sha256"]:
                raise ValueError("Corpus source hash changed: " + rec["id"])
            records.append({"id": rec["id"], "label": rec["label"], "prior_split": rec["split"], **checked})
        labeled = [r for r in records if r["label"] in {"failure", "clean"}]
        result = {"manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "validation_scope": "Development audit: prior heldout examples have been inspected; no untouched heldout claim.",
            "summary": {"recordings": len(records),
                "confirmed_failures": sum(r["label"] == "failure" for r in labeled),
                "noise_hits_on_confirmed_failures": sum(r["label"] == "failure" and r["strange_noise_suspected"] for r in labeled),
                "confirmed_clean": sum(r["label"] == "clean" for r in labeled),
                "noise_flags_on_confirmed_clean": sum(r["label"] == "clean" and r["strange_noise_suspected"] for r in labeled),
                "silence_flags_on_confirmed_clean": sum(r["label"] == "clean" and r["long_silence_suspected"] for r in labeled)},
            "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps(result.get("summary", {k: result[k] for k in ("status", "strange_noise_suspected", "long_silence_suspected") if k in result})))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
