"""Compare direct qwentts-cpp-python streaming with QwenEngine.

This benchmark intentionally requires an output directory outside the source
tree. It loads the native model twice sequentially (direct binding, then
RealtimeTTS) so the measurements do not compete for VRAM.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from RealtimeTTS import QwenEngine, QwenVoice
from RealtimeTTS.engines.qwen_engine import (
    DEFAULT_MODEL,
    SAMPLE_RATE,
    _float_to_pcm16,
    _load_reference_audio,
)


class RecordingQueue(queue.Queue):
    def __init__(self) -> None:
        super().__init__()
        self.arrivals: list[tuple[int, bytes]] = []

    def put(self, item, block=True, timeout=None):
        self.arrivals.append((time.perf_counter_ns(), item))
        return super().put(item, block=block, timeout=timeout)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(runs)}
    for field in (
        "wall_ms", "first_native_callback_ms", "first_audio_ms", "first_audible_ms",
        "rtf", "callback_adapter_ms",
    ):
        values = [float(run[field]) for run in runs if run.get(field) is not None]
        if values:
            result[field] = {
                "median": statistics.median(values),
                "mean": statistics.fmean(values),
                "p95": _percentile(values, 0.95),
            }
    return result


def _environment() -> dict[str, Any]:
    gpu = None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        gpu = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.TimeoutExpired):
        pass
    versions = {}
    for package in ("realtimetts", "qwentts-cpp-python", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "source-checkout"
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "gpu": gpu,
        "packages": versions,
    }


def _write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)


def _first_audible_arrival(
    arrivals: list[tuple[int, bytes]], started_ns: int, threshold: int
) -> float | None:
    playback_cursor_ns: int | None = None
    for arrival_ns, pcm in arrivals:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
        chunk_start_ns = max(arrival_ns, playback_cursor_ns or arrival_ns)
        audible = np.flatnonzero(np.abs(samples) >= threshold)
        if audible.size:
            return (
                (chunk_start_ns - started_ns) / 1_000_000
                + (int(audible[0]) / SAMPLE_RATE) * 1000
            )
        playback_cursor_ns = chunk_start_ns + int(samples.size / SAMPLE_RATE * 1_000_000_000)
    return None


def _direct_once(backend, kwargs: dict[str, Any], audible_threshold: int) -> tuple[dict[str, Any], bytes]:
    started_ns = time.perf_counter_ns()
    arrivals: list[tuple[int, bytes]] = []
    for chunk, sample_rate in backend.stream(cancel_event=threading.Event(), **kwargs):
        if int(sample_rate) != SAMPLE_RATE:
            raise RuntimeError(f"Direct binding returned {sample_rate} Hz, expected {SAMPLE_RATE}")
        pcm = _float_to_pcm16(chunk)
        if pcm:
            arrivals.append((time.perf_counter_ns(), pcm))
    ended_ns = time.perf_counter_ns()
    pcm = b"".join(chunk for _, chunk in arrivals)
    duration_s = len(pcm) / 2 / SAMPLE_RATE
    profile = dict(backend.last_stream_profile or {})
    first_audio_ms = (arrivals[0][0] - started_ns) / 1_000_000 if arrivals else None
    return {
        "wall_ms": (ended_ns - started_ns) / 1_000_000,
        "first_native_callback_ms": profile.get("first_callback_enter_ms"),
        "first_audio_ms": first_audio_ms,
        "first_audible_ms": _first_audible_arrival(arrivals, started_ns, audible_threshold),
        "audio_duration_s": duration_s,
        "rtf": ((ended_ns - started_ns) / 1_000_000_000) / duration_s if duration_s else None,
        "callback_adapter_ms": profile.get("first_callback_to_yield_ms"),
        "native_profile": profile,
    }, pcm


def _engine_once(
    engine: QwenEngine, text: str, audible_threshold: int
) -> tuple[dict[str, Any], bytes]:
    recording_queue = RecordingQueue()
    engine.queue = recording_queue
    started_ns = time.perf_counter_ns()
    if not engine.synthesize(text):
        raise RuntimeError(f"QwenEngine synthesis failed: {engine.last_error}")
    ended_ns = time.perf_counter_ns()
    pcm = b"".join(chunk for _, chunk in recording_queue.arrivals)
    duration_s = len(pcm) / 2 / SAMPLE_RATE
    profile = dict(engine.last_synthesis_profile)
    native_profile = dict(profile.get("native") or {})
    return {
        "wall_ms": (ended_ns - started_ns) / 1_000_000,
        "first_native_callback_ms": native_profile.get("first_callback_enter_ms"),
        "first_audio_ms": (
            (recording_queue.arrivals[0][0] - started_ns) / 1_000_000
            if recording_queue.arrivals else None
        ),
        "first_audible_ms": _first_audible_arrival(
            recording_queue.arrivals, started_ns, audible_threshold
        ),
        "audio_duration_s": duration_s,
        "rtf": ((ended_ns - started_ns) / 1_000_000_000) / duration_s if duration_s else None,
        "callback_adapter_ms": profile.get("callback_to_queue_ms"),
        "engine_profile": profile,
    }, pcm


def _common_stream_kwargs(args, voice_ref, *, icl: bool) -> dict[str, Any]:
    return {
        "text": args.text,
        "lang": args.language.lower(),
        "ref_spk_emb": voice_ref.ref_spk_emb,
        "ref_codes": voice_ref.ref_codes if icl else None,
        "ref_text": args.ref_text if icl else None,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }


def _run_direct(args, output_dir: Path) -> dict[str, Any]:
    import qwentts_cpp

    load_started = time.perf_counter()
    native_kwargs = {
        "library_path": args.library_path,
        "max_batch": 1,
        "codec_chunk_sec": args.codec_chunk_sec,
    }
    if args.talker_path and args.codec_path:
        backend = qwentts_cpp.QwenTTS(
            args.talker_path,
            args.codec_path,
            **native_kwargs,
        )
    else:
        backend = qwentts_cpp.QwenTTS.from_pretrained(
            model_id=args.model,
            quant=args.quant,
            cache_dir=args.model_cache_dir,
            local_files_only=args.local_files_only,
            **native_kwargs,
        )
    load_ms = (time.perf_counter() - load_started) * 1000
    try:
        voice_started = time.perf_counter()
        if args.spk_path and args.rvq_path:
            voice_ref = backend.load_voice_ref(args.spk_path, args.rvq_path)
        else:
            voice_ref = backend.extract_voice_ref(_load_reference_audio(args.ref_audio))
        voice_prepare_ms = (time.perf_counter() - voice_started) * 1000
        modes = {}
        for mode, icl in (("x_vector", False), ("icl", True)):
            kwargs = _common_stream_kwargs(args, voice_ref, icl=icl)
            cold, _ = _direct_once(backend, kwargs, args.audible_threshold)
            # A separate unmeasured request stabilizes graph/cache state.
            _direct_once(backend, kwargs, args.audible_threshold)
            warm = []
            representative_sha256 = None
            for index in range(args.runs):
                run, pcm = _direct_once(backend, kwargs, args.audible_threshold)
                run["index"] = index + 1
                warm.append(run)
                if index == 0:
                    _write_wav(output_dir / f"direct_{mode}.wav", pcm)
                    representative_sha256 = hashlib.sha256(pcm).hexdigest()
            modes[mode] = {
                "first_request": cold,
                "warm_runs": warm,
                "summary": _summary(warm),
                "representative_pcm_sha256": representative_sha256,
            }
        return {"load_ms": load_ms, "voice_prepare_ms": voice_prepare_ms, "modes": modes}
    finally:
        backend.close()


def _run_engine(args, output_dir: Path) -> dict[str, Any]:
    load_started = time.perf_counter()
    engine = QwenEngine(
        model_id=args.model,
        quant=args.quant,
        talker_path=args.talker_path,
        codec_path=args.codec_path,
        model_cache_dir=args.model_cache_dir,
        voice_cache_dir=output_dir / "voice-cache",
        local_files_only=args.local_files_only,
        library_path=args.library_path,
        max_batch=1,
        codec_chunk_sec=args.codec_chunk_sec,
        warmup=False,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    load_ms = (time.perf_counter() - load_started) * 1000
    try:
        modes = {}
        voice_prepare_ms = None
        for mode, ref_text in (("x_vector", None), ("icl", args.ref_text)):
            voice_started = time.perf_counter()
            engine.set_voice(
                QwenVoice(
                    mode,
                    ref_audio=args.ref_audio if not args.spk_path else None,
                    ref_text=ref_text,
                    language=args.language,
                    spk_path=args.spk_path,
                    rvq_path=args.rvq_path,
                )
            )
            if voice_prepare_ms is None:
                voice_prepare_ms = (time.perf_counter() - voice_started) * 1000
            cold, _ = _engine_once(engine, args.text, args.audible_threshold)
            _engine_once(engine, args.text, args.audible_threshold)
            warm = []
            representative_sha256 = None
            for index in range(args.runs):
                run, pcm = _engine_once(engine, args.text, args.audible_threshold)
                run["index"] = index + 1
                warm.append(run)
                if index == 0:
                    _write_wav(output_dir / f"engine_{mode}.wav", pcm)
                    representative_sha256 = hashlib.sha256(pcm).hexdigest()
            modes[mode] = {
                "first_request": cold,
                "warm_runs": warm,
                "summary": _summary(warm),
                "representative_pcm_sha256": representative_sha256,
            }
        return {"load_ms": load_ms, "voice_prepare_ms": voice_prepare_ms, "modes": modes}
    finally:
        engine.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--spk-path", type=Path)
    parser.add_argument("--rvq-path", type=Path)
    parser.add_argument("--ref-text", required=True, help="Exact reference transcript for ICL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text", default="This is a reproducible Qwen latency benchmark.")
    parser.add_argument("--language", default="english")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quant", default="Q8_0")
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--talker-path", type=Path)
    parser.add_argument("--codec-path", type=Path)
    parser.add_argument("--library-path", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--codec-chunk-sec", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--audible-threshold", type=int, default=256)
    args = parser.parse_args()
    args.ref_text = args.ref_text.strip()
    if not args.ref_text:
        parser.error("--ref-text must not be empty")
    if bool(args.talker_path) != bool(args.codec_path):
        parser.error("--talker-path and --codec-path must be supplied together")
    if bool(args.spk_path) != bool(args.rvq_path):
        parser.error("--spk-path and --rvq-path must be supplied together")
    if not args.ref_audio and not args.spk_path:
        parser.error("supply --ref-audio or a --spk-path/--rvq-path pair")
    if args.runs < 30:
        parser.error("--runs must be at least 30 for the acceptance benchmark")
    output_dir = args.output_dir.expanduser().resolve()
    source_root = Path(__file__).resolve().parents[1]
    if output_dir == source_root or source_root in output_dir.parents:
        parser.error("--output-dir must be outside the RealtimeTTS repository")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": 1,
        "model": args.model,
        "quant": args.quant,
        "text": args.text,
        "language": args.language,
        "runs_per_mode": args.runs,
        "sample_rate": SAMPLE_RATE,
        "environment": _environment(),
        "direct": _run_direct(args, output_dir),
        "engine": _run_engine(args, output_dir),
    }
    acceptance = {}
    for mode in ("x_vector", "icl"):
        direct_rtf = report["direct"]["modes"][mode]["summary"]["rtf"]["median"]
        engine_rtf = report["engine"]["modes"][mode]["summary"]["rtf"]["median"]
        adapter_ms = report["engine"]["modes"][mode]["summary"]["callback_adapter_ms"]["median"]
        direct_pcm_hash = report["direct"]["modes"][mode]["representative_pcm_sha256"]
        engine_pcm_hash = report["engine"]["modes"][mode]["representative_pcm_sha256"]
        acceptance[mode] = {
            "callback_to_queue_le_10ms": adapter_ms <= 10.0,
            "engine_rtf_within_15_percent": engine_rtf <= direct_rtf * 1.15,
            "engine_rtf_le_0_25": engine_rtf <= 0.25,
            "representative_pcm_identical": direct_pcm_hash == engine_pcm_hash,
            "direct_median_rtf": direct_rtf,
            "engine_median_rtf": engine_rtf,
            "callback_to_queue_median_ms": adapter_ms,
        }
    report["acceptance"] = acceptance
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)
    return 0 if all(all(v for k, v in mode.items() if isinstance(v, bool)) for mode in acceptance.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
