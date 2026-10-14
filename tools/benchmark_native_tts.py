"""Capture uncensored native CPU TTS in an isolated process.

Uses existing model files and registered voices read-only. No live service is
changed. Records native float PCM before engine processing, queued PCM16,
arrival times, exact settings, source hashes and native generation profiles.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import queue
import sys
import time
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def save_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voice-dir", type=Path, required=True)
    parser.add_argument("--talker", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    cases = json.loads(args.requests.read_text(encoding="utf-8"))["cases"]
    registry = json.loads((args.voice_dir / "manifest.json").read_text(encoding="utf-8"))
    records = {record["name"]: record for record in registry["voices"]}
    for case in cases:
        record = records[case["voice"]]
        if record["kind"] not in {"wav", "latents"}:
            raise ValueError("Unsupported registered voice format")
        for relative, expected in zip(record["files"], record["sha256"]):
            path = (args.voice_dir / relative).resolve()
            path.relative_to(args.voice_dir.resolve())
            if digest(path) != expected:
                raise ValueError("Registered reference hash mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    from RealtimeTTS import QwenVoice
    from RealtimeTTS.engines.qwen_cpu_engine import QwenCpuEngine

    class CapturedEngine(QwenCpuEngine):
        def _stream_native(self, stream_kwargs, *, stream_started_ns=None):
            for audio, rate in super()._stream_native(
                    stream_kwargs, stream_started_ns=stream_started_ns):
                audio = np.asarray(audio, dtype=np.float32).copy()
                self.native_chunks.append(audio)
                self.native_arrivals.append({
                    "time_ms": (time.perf_counter() - self.case_started) * 1000,
                    "samples": len(audio), "sample_rate": rate})
                yield audio, rate

    class CapturedQueue(queue.Queue):
        def put(self, value, *pos, **kw):
            engine.queue_arrivals.append({
                "time_ms": (time.perf_counter() - engine.case_started) * 1000,
                "samples": len(value) // 2})
            return super().put(value, *pos, **kw)

    settings = dict(talker_path=args.talker, codec_path=args.codec,
                    quant="Q8_0", local_files_only=True,
                    cpu_threads=7, cpu_codec_threads=7, cpu_stream_frames=2,
                    cpu_affinity=0x5545, cpu_codec_affinity=0xaa8a,
                    clamp_fp16=False, clone_mode="speaker_only",
                    startup_buffer_ms=80, warmup=False,
                    hallucination_guard=False, streaming_aligner=None,
                    voice_cache_dir=args.output / "voice-cache",
                    max_new_tokens=args.max_new_tokens, seed=20261009)
    engine = CapturedEngine(**settings)
    engine.queue = CapturedQueue()
    assert engine.streaming_aligner is None and not engine.hallucination_guard
    if hasattr(engine._backend, "set_log_callback"):
        engine._backend.set_log_callback(lambda level, message: None)
    report = {
        "kind": "uncensored_native_cpu_tts",
        "guard_enabled": False, "source_root": str(ROOT),
        "model": {"talker": str(args.talker), "talker_sha256": digest(args.talker),
                  "codec": str(args.codec), "codec_sha256": digest(args.codec),
                  "native_version": engine.native_version,
                  "native_abi": engine.native_abi_version},
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in settings.items()},
        "requests_sha256": digest(args.requests),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p)
                          for p in sorted((ROOT / "RealtimeTTS").rglob("*.py"))},
        "cases": []}
    try:
        for index, case in enumerate(cases):
            record = records[case["voice"]]
            voice_paths = [args.voice_dir / value for value in record["files"]]
            voice_options = ({"ref_audio": voice_paths[0]} if record["kind"] == "wav"
                             else {"spk_path": voice_paths[0], "rvq_path": voice_paths[1]})
            voice = QwenVoice(case["voice"], **voice_options,
                              ref_text=record.get("ref_text") or None,
                              language=case["language"])
            engine.set_voice(voice)
            engine.seed = int(case.get("seed", 20261009 + index))
            engine.native_chunks, engine.native_arrivals = [], []
            engine.queue_arrivals = []
            engine.case_started = time.perf_counter()
            ok = engine.synthesize(case["text"])
            elapsed = (time.perf_counter() - engine.case_started) * 1000
            if not ok or engine.last_error:
                raise RuntimeError(str(engine.last_error))
            chunks = []
            while not engine.queue.empty():
                chunks.append(engine.queue.get_nowait())
            pcm = np.frombuffer(b"".join(chunks), dtype="<i2")
            native = np.concatenate(engine.native_chunks)
            if not len(pcm) or not np.any(pcm) or not np.isfinite(native).all():
                raise ValueError("Empty, silent or nonfinite generated audio")
            rate = 24000
            name = case["id"] + ".wav"
            native_name = case["id"] + ".native.wav"
            sf.write(args.output / name, pcm, rate, subtype="PCM_16")
            sf.write(args.output / native_name, native, rate, subtype="FLOAT")
            item = {**case, "audio": name, "native_audio": native_name,
                    "seed": engine.seed, "duration_seconds": len(pcm) / rate,
                    "native_duration_seconds": len(native) / rate,
                    "audio_sha256": digest(args.output / name),
                    "native_audio_sha256": digest(args.output / native_name),
                    "arrivals": engine.queue_arrivals,
                    "native_arrivals": engine.native_arrivals,
                    "profile": engine.last_synthesis_profile,
                    "wall_ms": elapsed, "voice_reference_sha256": record["sha256"]}
            save_json(args.output / (case["id"] + ".json"), item)
            report["cases"].append(item)
            print(json.dumps({"id": case["id"], "duration_seconds": len(pcm) / rate,
                              "native_seconds": len(native) / rate,
                              "wall_ms": elapsed}), flush=True)
    finally:
        engine.shutdown()
    save_json(args.output / "manifest.json", report)
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"])}))

if __name__ == "__main__":
    main()
