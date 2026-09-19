"""Measure live PCM delivery, including the startup reserve needed to avoid gaps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import tomllib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RealtimeTTS.engines.prefix_splice_engine import QwenHttpSynthesizer


def measure(backend, text):
    start = time.perf_counter()
    delivered = 0
    arrivals = []
    first_active = None
    for chunk in backend.stream(text):
        elapsed = time.perf_counter() - start
        if first_active is None and np.any(np.abs(chunk) >= .005):
            first_active = elapsed
        arrivals.append({"at_seconds": elapsed, "prior_audio_seconds": delivered / backend.sample_rate,
                         "samples": len(chunk)})
        delivered += len(chunk)
    if not arrivals:
        raise RuntimeError("Server returned no PCM")
    # Offline lower bound for a continuously paced sink, using observed delivery.
    required_start = max(x["at_seconds"] - x["prior_audio_seconds"] for x in arrivals)
    return {"text": text, "first_pcm_ms": arrivals[0]["at_seconds"] * 1000,
            "first_active_packet_ms": None if first_active is None else first_active * 1000,
            "minimum_gapless_start_ms": required_start * 1000,
            "duration_seconds": delivered / backend.sample_rate,
            "wall_seconds": time.perf_counter() - start, "arrivals": arrivals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--urls", nargs="+", default=["http://192.168.178.22:18084", "http://192.168.178.22:18085"])
    parser.add_argument("--voice", default="mira_v5_spark_de")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key = tomllib.loads(args.config.read_text(encoding="utf-8"))["tts"]["api_key"]
    report = {"note": "API delivery, not physical playback. Gapless start is a hindsight lower bound, not a future guarantee.", "servers": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for url in args.urls:
        server = {"url": url, "runs": []}
        report["servers"].append(server)
        backend = QwenHttpSynthesizer(url, key, voice=args.voice, timeout=15)
        try:
            backend.prepare()
            server["capabilities"] = backend.capabilities
            print(json.dumps({"url": url, "capabilities": backend.capabilities}, ensure_ascii=False), flush=True)
            measure(backend, "Heute Morgen")
            texts = ["Heute Morgen", "Heute Morgen gehen wir gemeinsam durch den Park und genießen die frische Luft."]
            for repetition in range(args.repeats):
                for text in texts[::1 if repetition % 2 == 0 else -1]:
                    run = measure(backend, text)
                    server["runs"].append(run)
                    print(json.dumps({k: v for k, v in run.items() if k != "arrivals"}, ensure_ascii=False), flush=True)
        except Exception as error:
            server["error"] = f"{type(error).__name__}: {error}"
            print(f"{url}: {server['error']}", flush=True)
        finally:
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
