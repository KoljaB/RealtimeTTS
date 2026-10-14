"""Reproducible CPU alignment checks against a local, provenance-bearing manifest.

Manifest: {"cases": [{"id": "...", "audio": "...wav/flac", "text": "...",
"words": [{"word": "...", "start_time": 0.1, "end_time": 0.3}]}]}.
Word labels are optional; transcript-only cases never produce boundary metrics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from RealtimeTTS.alignment.backends import CPUAlignmentModel
from RealtimeTTS.alignment.onnx_ctc import OnnxCTCBackend
from RealtimeTTS.alignment.streaming import StreamingTextAligner
from RealtimeTTS.alignment.evidence import CompletionPolicy


def reference_aligner(ref):
    source = subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "show",
         f"{ref}:RealtimeTTS/alignment/streaming.py"],
        cwd=ROOT, text=True, encoding="utf-8",
    )
    module = types.ModuleType("RealtimeTTS.alignment._benchmark_baseline")
    module.__package__ = "RealtimeTTS.alignment"
    exec(compile(source, f"{ref}:streaming.py", "exec"), module.__dict__)
    return module.StreamingTextAligner


def stream_probe(aligner_class, backend, audio, rate, text):
    events = []
    aligner = aligner_class(backend=backend)
    session = aligner.start(text, rate, callback=events.append)
    hop = round(aligner.interval_seconds * rate)
    padded = np.pad(audio, (0, round(.64 * rate)))
    padded = np.pad(padded, (0, (-len(padded)) % hop))
    started = time.perf_counter()
    try:
        for end in range(hop, len(padded) + 1, hop):
            session.feed(padded[end - hop:end])
            deadline = time.monotonic() + 15
            while session.analyzed_seconds + 1e-6 < end / rate and not session.done.is_set():
                if time.monotonic() >= deadline:
                    raise TimeoutError("alignment did not consume the submitted audio prefix")
                session.done.wait(.001)
            if session.done.is_set():
                break
        session.finish()
        if not session.done.wait(15):
            raise TimeoutError("alignment did not finalize")
        complete = next((e for e in events if e["type"] == "alignment_complete"), {})
        return {
            "stop_requested": session.stop_requested.is_set(),
            "complete": session.complete, "error": session.error,
            "finish_reason": session.finish_reason,
            "end_time": complete.get("end_time"),
            "trim_end_time": complete.get("trim_end_time"),
            "detection_audio_seconds": session.analyzed_seconds,
            "inference_calls": session.inference_calls,
            "inference_ms": session.inference_ms,
            "wall_ms": (time.perf_counter() - started) * 1000,
        }
    finally:
        session.finish(cancelled=True)
        aligner.close()


def variants(case, audio, rate, *, stress=False):
    yield "full", audio, case["text"], True
    labels = case.get("words")
    cut = int(labels[-1]["start_time"] * rate) if labels else int(.6 * len(audio))
    yield "missing_suffix", audio[:cut], case["text"], False
    yield "wrong_text", audio, "please open the window and bring me a glass of water", False
    yield "extra_speech", np.concatenate((audio, np.zeros(round(.3 * rate)), audio)), case["text"], True
    if stress:
        words = case["text"].split()
        replacement = "garden" if words[-1].lower().strip(".!?") == "water" else "water"
        yield "wrong_final_word", audio, " ".join(words[:-1] + [replacement]), False
        rng = np.random.default_rng(20261009)
        noise = rng.normal(size=len(audio)).astype(np.float32)
        noise *= np.sqrt(np.mean(audio ** 2) / (10 * np.mean(noise ** 2)))
        yield "noise_10db", audio + noise, case["text"], True
        yield "silence", np.zeros(min(len(audio), rate * 3), dtype=np.float32), case["text"], False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-ref", help="Trusted local Git revision for paired stop-policy checks")
    parser.add_argument("--stress", action="store_true",
                        help="Also check changed final words, 10 dB additive noise, and silence")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new result file")
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))["cases"]
    backend = OnnxCTCBackend(args.model, language="english")
    model = CPUAlignmentModel(backend, "english")
    versions = {"current": StreamingTextAligner}
    if args.baseline_ref:
        versions["baseline"] = reference_aligner(args.baseline_ref)
    report = {
        "platform": platform.platform(), "threads": 2,
        "model_sha256": hashlib.sha256((args.model / "model_quantized.onnx").read_bytes()).hexdigest(),
        "baseline_ref": args.baseline_ref, "cases": [],
        "completion_policy": asdict(CompletionPolicy()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("RealtimeTTS/alignment/evidence.py",
                                       "RealtimeTTS/alignment/streaming.py")},
        "limits": "Small public CPU speech probe; not TTS end-to-end latency or broad calibration.",
    }
    # Warm both controller paths against identical short acoustic input.
    audio, rate = sf.read(cases[0]["audio"], dtype="float32")
    for cls in versions.values():
        stream_probe(cls, backend, audio, rate, cases[0]["text"])
    pair = 0
    for case in cases:
        audio, rate = sf.read(case["audio"], dtype="float32")
        if audio.ndim != 1:
            raise ValueError("benchmark requires mono reference audio")
        record = {"id": case["id"], "source": case.get("source"),
                  "audio_sha256": hashlib.sha256(Path(case["audio"]).read_bytes()).hexdigest(),
                  "duration_seconds": len(audio) / rate, "probes": []}
        alignment = model.align(case["text"], audio, rate)
        record["aligned_words"] = len(alignment["words"])
        labels = case.get("words")
        if labels:
            predicted = alignment["words"]
            if [w["word"] for w in predicted] == [w["word"] for w in labels]:
                errors = [abs(p[key] - g[key]) * 1000
                          for p, g in zip(predicted, labels)
                          for key in ("start_time", "end_time")]
                record["boundary_error_ms"] = {
                    "median": float(np.median(errors)), "p95": float(np.percentile(errors, 95)),
                    "max": max(errors), "boundaries": len(errors),
                }
            else:
                record["boundary_error_unavailable"] = "aligned words differ from the reference"
        for name, samples, text, expected in variants(case, audio, rate, stress=args.stress):
            probe = {"variant": name, "expected_stop": expected, "runs": {}}
            order = list(versions)
            if pair % 2:
                order.reverse()
            pair += 1
            for version in order:
                result = stream_probe(versions[version], backend, samples, rate, text)
                result["matches_expected_stop"] = result["stop_requested"] == expected
                if name == "extra_speech":
                    onset = len(audio) / rate + .3
                    result["extra_speech_onset_seconds"] = onset
                    result["detected_before_extra_speech"] = (
                        result["stop_requested"] and result["detection_audio_seconds"] <= onset)
                    result["trim_before_extra_speech"] = (
                        result["trim_end_time"] is not None and result["trim_end_time"] <= onset)
                if labels and expected and result["trim_end_time"] is not None:
                    result["premature_trim"] = result["trim_end_time"] < labels[-1]["end_time"]
                probe["runs"][version] = result
            record["probes"].append(probe)
            print(json.dumps({"id": case["id"], "variant": name,
                              "stops": {k: v["stop_requested"] for k, v in probe["runs"].items()}}),
                  flush=True)
        report["cases"].append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"])}))


if __name__ == "__main__":
    main()
