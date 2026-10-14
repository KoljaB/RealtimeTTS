"""Capture real TTS PCM, then replay it through QwenEngine and real alignment.

Capture keeps the server's existing settings. Replay changes no live service:
the recorded waveform is injected at the native PCM seam, at its original
arrival times. This exercises the current queue, guard, fade and cancellation
path, but does not measure hardware playback or an updated remote deployment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def capture(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    headers = {}
    key = os.environ.get(args.api_key_env)
    if key:
        headers["Authorization"] = "Bearer " + key
    base = args.base_url.rstrip("/")
    with requests.Session() as client:
        response = client.get(base + "/v1/capabilities", headers=headers, timeout=20)
        response.raise_for_status()
        capabilities = response.json()
        report = {"kind": "real_tts_capture", "base_url": base,
                  "capabilities": capabilities, "cases": []}
        for case in json.loads(args.manifest.read_text(encoding="utf-8"))["cases"]:
            started = time.perf_counter()
            payload = {k: case[k] for k in ("voice", "language", "text")}
            payload["input"] = payload.pop("text")
            payload.update(response_format="pcm", seed=20261009, max_new_tokens=512)
            chunks, arrivals, pending = [], [], b""
            with client.post(base + "/v1/audio/speech", json=payload, headers=headers,
                             stream=True, timeout=(10, 180)) as response:
                response.raise_for_status()
                rate = int(response.headers.get("X-Audio-Sample-Rate", 24000))
                if rate != 24000:
                    raise ValueError("Qwen replay requires mono 24 kHz PCM16")
                for block in response.iter_content(chunk_size=None):
                    pending += block
                    usable = len(pending) - len(pending) % 2
                    if not usable:
                        continue
                    chunks.append(pending[:usable])
                    pending = pending[usable:]
                    arrivals.append({"time_ms": (time.perf_counter() - started) * 1000,
                                     "samples": usable // 2})
            if pending or not chunks:
                raise ValueError("empty or incomplete PCM response")
            samples = np.frombuffer(b"".join(chunks), dtype="<i2")
            if not np.any(samples):
                raise ValueError("TTS produced only silence")
            audio_path = output / (case["id"] + ".wav")
            sf.write(audio_path, samples, rate, subtype="PCM_16")
            record = {**case, "audio": str(audio_path.resolve()), "sample_rate": rate,
                      "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                      "duration_seconds": len(samples) / rate, "arrivals": arrivals,
                      "request_wall_ms": (time.perf_counter() - started) * 1000,
                      "first_pcm_ms": arrivals[0]["time_ms"]}
            report["cases"].append(record)
            write_json(output / (case["id"] + ".json"), record)
            print(json.dumps({k: record[k] for k in ("id", "duration_seconds", "first_pcm_ms")}),
                  flush=True)
    write_json(output / "manifest.json", report)


class RecordedPCM:
    # The adapter exposes one recording as a named source. It loads no TTS model.
    model_type = "custom_voice"

    def __init__(self, audio, arrivals, extra_tail):
        self.audio, self.arrivals, self.extra_tail = audio, arrivals, extra_tail
        self.cancel_observed = False
        self.produced_samples = 0

    def speaker_names(self):
        return ["recording"]

    def close(self):
        pass

    def stream(self, *, cancel_event, **kwargs):
        started, offset = time.perf_counter(), 0
        for arrival in self.arrivals:
            delay = arrival["time_ms"] / 1000 - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)
            if cancel_event.is_set():
                self.cancel_observed = True
                return
            count = arrival["samples"]
            chunk = self.audio[offset:offset + count]
            offset += count
            self.produced_samples += len(chunk)
            yield chunk, 24000
        if self.extra_tail:
            # A controlled repeat models surplus generated speech. Its onset is
            # known exactly; this is not a claim of naturally hallucinated TTS.
            tail = np.r_[np.zeros(7200, dtype=np.float32), self.audio]
            for offset in range(0, len(tail), 3840):
                if cancel_event.is_set():
                    self.cancel_observed = True
                    return
                chunk = tail[offset:offset + 3840]
                time.sleep(len(chunk) / 24000)
                self.produced_samples += len(chunk)
                yield chunk, 24000


def replay(args):
    from RealtimeTTS import QwenEngine, QwenVoice
    from RealtimeTTS.alignment.streaming import StreamingTextAligner
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    aligner = StreamingTextAligner(
        model_dir=args.model, language="english",
        multilingual_model=args.multilingual_model)
    report = {"kind": "real_pcm_engine_replay",
              "completion_policy": asdict(aligner.completion_policy),
              "capture_manifest": str(args.manifest.resolve()), "cases": [],
              "capture_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("RealtimeTTS/alignment/evidence.py",
                                             "RealtimeTTS/alignment/streaming.py",
                                             "RealtimeTTS/engines/qwen_engine.py")},
              "limits": "Recorded native PCM seam; not hardware playback or remote deployment."}
    report["startup_buffer_ms"] = args.startup_buffer_ms
    report["warmup_cases"] = []
    try:
        if args.warmup_manifest:
            warmups = json.loads(args.warmup_manifest.read_text(encoding="utf-8"))["cases"]
            for warmup in warmups:
                samples, rate = sf.read(warmup["audio"], dtype="float32")
                session = aligner.start(warmup["text"], rate, language=warmup["language"])
                session.feed(samples)
                session.finish()
                if not session.done.wait(30) or session.error:
                    raise RuntimeError("Alignment warmup failed: " + str(session.error))
                report["warmup_cases"].append(warmup["id"])
        for case_index, case in enumerate(json.loads(args.manifest.read_text(encoding="utf-8"))["cases"]):
            audio, rate = sf.read(case["audio"], dtype="float32")
            if rate != 24000 or audio.ndim != 1:
                raise ValueError("expected mono 24 kHz capture")
            record = {"id": case["id"], "text": case["text"],
                      "language": case["language"], "duration_seconds": len(audio) / rate,
                      "expected_completion": case.get("expected_completion", True),
                      "runs": {}}
            paired = ("natural", "unaligned") if case_index % 2 else ("unaligned", "natural")
            for variant in (*paired, "extra_speech"):
                backend = RecordedPCM(audio, case["arrivals"], variant == "extra_speech")
                events = []
                started = time.perf_counter()
                def event(value):
                    events.append({**value, "wall_ms": (time.perf_counter() - started) * 1000})
                engine = QwenEngine(
                    voice=QwenVoice("recording", speaker="recording", language=case["language"]),
                    voice_cache_dir=output / "voice-cache", backend_factory=lambda **kw: backend,
                    streaming_aligner=None if variant == "unaligned" else aligner,
                    hallucination_guard=variant != "unaligned", on_alignment=event, warmup=False,
                    trim_silence=False, startup_buffer_ms=args.startup_buffer_ms)
                try:
                    ok = engine.synthesize(case["text"])
                    if engine.last_alignment is not None and not engine.last_alignment.done.wait(30):
                        raise TimeoutError("alignment failed to finish after real PCM replay")
                    pcm = []
                    while not engine.queue.empty():
                        pcm.append(engine.queue.get_nowait())
                    pcm = b"".join(pcm)
                    sf.write(output / (case["id"] + "-" + variant + ".wav"),
                             np.frombuffer(pcm, dtype="<i2"), rate, subtype="PCM_16")
                    complete = next((e for e in events if e["type"] == "alignment_complete"), None)
                    record["runs"][variant] = {
                        "ok": ok, "error": str(engine.last_error) if engine.last_error else None,
                        "events": events, "profile": engine.last_synthesis_profile,
                        "output_seconds": len(pcm) / 2 / rate,
                        "produced_seconds": backend.produced_samples / rate,
                        "extra_speech_delivered_seconds": (
                            max(0, len(pcm) / 2 / rate - len(audio) / rate - .3)
                            if backend.extra_tail else 0),
                        "cancel_observed": backend.cancel_observed,
                        "completion": complete,
                        "extra_speech_onset_seconds": len(audio) / rate + .3 if backend.extra_tail else None,
                        "wall_ms": (time.perf_counter() - started) * 1000}
                    print(json.dumps({"id": case["id"], "variant": variant, "ok": ok,
                                      "verified": bool(complete),
                                      "output_seconds": len(pcm) / 2 / rate}), flush=True)
                finally:
                    engine.shutdown()
            report["cases"].append(record)
            write_json(output / (case["id"] + ".json"), record)
    finally:
        aligner.close()
    write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "replay"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="REALTIME_TTS_API_KEY")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--multilingual-model", type=Path)
    parser.add_argument("--warmup-manifest", type=Path)
    parser.add_argument("--startup-buffer-ms", type=int, default=80)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new directory")
    if args.mode == "capture" and not args.base_url:
        parser.error("capture requires --base-url")
    if args.mode == "replay" and not args.model:
        parser.error("replay requires --model")
    (capture if args.mode == "capture" else replay)(args)


if __name__ == "__main__":
    main()
