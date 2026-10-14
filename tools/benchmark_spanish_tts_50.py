r"""Reproducible 50/100-case Qwen CPU TTS audit, with offline CUDA ASR.

Run with C:\Python\Python312\python.exe -B -S. No cloud calls, playback,
voice creation, service changes, retries, trimming or automatic cancellation.
"""
from __future__ import annotations
import argparse
import asyncio
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import tomllib
import unicodedata
from urllib.parse import urlsplit
import wave
import annotate_tts_faster_whisper as fw

CORPUS = Path(__file__).parent / "data" / "spanish_tts_50.json"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, data):
    # Task-owned checkpoints; Windows replace is atomic for readers.
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)

def normalized_text(text):
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch) and ch.isalnum())


def validate_corpus(corpus):
    entries = corpus["entries"]
    total = len(entries)
    ids = {e["id"] for e in entries}
    short = [e for e in entries if e["class"] == "short"]
    longer = [e for e in entries if e["class"] == "longer"]
    expected_shape = {50: (40, 10), 100: (90, 10)}.get(total)
    if expected_shape is None or len(ids) != total:
        raise ValueError("Supported corpora require 50 or 100 unique cases")
    if (len(short), len(longer)) != expected_shape:
        raise ValueError("Corpus must contain 40/90 short and 10 longer cases")
    if [e.get("index") for e in entries] != list(range(1, total + 1)):
        raise ValueError("Corpus indices must be consecutive and ordered")
    normalized = [normalized_text(e["text"]) for e in entries]
    if len(set(normalized)) != total:
        raise ValueError("Corpus texts must be unique after normalization")
    declared = corpus.get("counts")
    if declared is not None:
        expected_counts = {"total": total, "short_1_to_3_words": len(short), "longer": len(longer)}
        if any(declared.get(key) != value for key, value in expected_counts.items()):
            raise ValueError("Corpus counts metadata does not match entries")
    for entry in entries:
        count = len(entry["text"].split())
        if count != entry["word_count"] or (entry["class"] == "short" and not 1 <= count <= 3):
            raise ValueError("Corpus word-count mismatch: " + entry["id"])
        if entry["class"] == "longer" and count <= 3:
            raise ValueError("Longer cases must contain more than three words")
    return entries

def resolve_key(cfg, values):
    key = cfg.get("api_key", "")
    if key.startswith(chr(36) + "{") and key.endswith("}"):
        key = os.environ.get(key[2:-1]) or values.get(key[2:-1], "")
    if not key:
        name = cfg.get("token_env", "")
        key = os.environ.get(name) or values.get(name, "")
    if not key:
        raise ValueError("TTS authentication is not configured")
    return key

def totals(rows, expected_total=50):
    complete = [r for r in rows if r.get("check") is not None]
    return {"requested": expected_total, "attempted": len(rows),
        "synthesis_complete": sum(r["tts"]["status"] == "complete" for r in rows),
        "synthesis_errors": sum(r["tts"]["status"] != "complete" for r in rows),
        "asr_complete": sum(r.get("asr_status") == "complete" for r in rows),
        "asr_errors": sum(r.get("asr_status") == "error" for r in rows),
        "noise_suspected": sum(r["check"]["strange_noise_suspected"] for r in complete),
        "long_silence": sum(r["check"]["long_silence_suspected"] for r in complete),
        "silence_only": sum(r["check"]["long_silence_suspected"] and not r["check"]["strange_noise_suspected"] for r in complete),
        "no_anomaly_found": sum(not r["check"]["strange_noise_suspected"] and not r["check"]["long_silence_suspected"]
            and r.get("asr_status") == "complete" and r["check"]["asr_status"] == "word_timestamps_available" for r in complete),
        "unclassified": sum(r.get("check") is None or r.get("asr_status") != "complete"
            or r["check"]["asr_status"] != "word_timestamps_available" for r in rows),
        "human_confirmed_hallucinations": None}

async def synthesize(entry, options, key, rate, timeout, directory):
    import websockets
    started = time.perf_counter()
    uri = options["endpoint"].replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + "/v1/audio/speech-stream"
    data, events, chunks = bytearray(), [], []
    first = None
    error = None
    done = False
    try:
        async with websockets.connect(uri, extra_headers={"Authorization": "Bearer " + key},
                open_timeout=min(timeout, 15), close_timeout=5, max_size=8 * 1024 * 1024) as ws:
            request = {k: v for k, v in options.items() if k != "endpoint"}
            await ws.send(json.dumps({"type": "config", **request}))
            await ws.send(json.dumps({"type": "text", "text": entry["text"]}, ensure_ascii=False))
            await ws.send(json.dumps({"type": "end"}))
            while not done:
                remaining = timeout - (time.perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("Case deadline")
                message = await asyncio.wait_for(ws.recv(), remaining)
                received = time.perf_counter() - started
                if isinstance(message, bytes):
                    if first is None:
                        first = received
                    chunks.append({"arrival_s": received, "offset_bytes": len(data), "bytes": len(message)})
                    data.extend(message)
                else:
                    event = json.loads(message)
                    events.append({"arrival_s": received, "event": event})
                    if event.get("sample_rate_hz", rate) != rate:
                        raise ValueError("Producer changed sample rate")
                    if event.get("type") == "error":
                        raise RuntimeError("TTS server error")
                    done = event.get("type") == "done"
    except Exception as exc:
        error = type(exc).__name__  # Credentials never appear in diagnostics.
    success = done and bool(data) and len(data) % 2 == 0 and error is None
    if not success and error is None:
        error = "MissingOrMalformedPCM"
    wav_path = directory / "wav" / (entry["id"] + (".wav" if success else ".partial.wav"))
    if data and len(data) % 2 == 0:
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(data)
    write_json(directory / "streams" / (entry["id"] + ".json"), {"events": events, "chunks": chunks})
    observed = [e["event"].get("seed") for e in events if e["event"].get("seed") is not None]
    return {"status": "complete" if success else "error", "error": error,
        "requested_seed": options["seed"], "observed_seeds": observed,
        "seed_evidence": "server_callback" if observed else "requested_supported_parameter_not_echoed",
        "wall_s": time.perf_counter() - started, "first_pcm_s": first,
        "wav_path": str(wav_path), "audio_s": len(data) / (rate * 2),
        "sha256": digest(wav_path) if wav_path.exists() else None}

def overview(directory, metadata, rows, elapsed, model_load_s=0, expected_total=50):
    count = totals(rows, expected_total)
    timing = {"scope_note":"Component sums include reused recorded work; total wall time is this invocation only.", "total_wall_s_this_invocation": elapsed,
        "synthesis_wall_s": sum(r["tts"]["wall_s"] for r in rows),
        "asr_model_load_s": model_load_s,
        "asr_inference_s": sum(r.get("asr_wall_s", 0) for r in rows),
        "detector_s": sum(r.get("check", {}).get("analysis_ms", 0) / 1000 for r in rows if r.get("check"))}
    result = {"status": "complete" if count["synthesis_complete"] == expected_total and count["asr_complete"] == expected_total else "incomplete",
        "metadata": metadata, "counts": count, "timing": timing, "rows": rows,
        "qualification": "Automatic suspicion, not human-confirmed labels. No anomaly found does not prove clean speech."}
    write_json(directory / "summary.json", result)
    columns = ["id", "class", "text", "words", "audio_s", "last_asr_word_end_s", "noise_suspected", "long_silence", "tts_s", "asr_s", "asr_status", "asr_error", "status", "wav"]
    with (directory / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            e, c = row["entry"], row.get("check") or {}
            writer.writerow(dict(id=e["id"], **{"class":e["class"]}, text=e["text"], words=e["word_count"],
                audio_s=row["tts"]["audio_s"], last_asr_word_end_s=(c.get("asr_evidence") or {}).get("last_word_end_s"),
                noise_suspected=c.get("strange_noise_suspected"), long_silence=c.get("long_silence_suspected"),
                tts_s=row["tts"]["wall_s"], asr_s=row.get("asr_wall_s"), asr_status=row.get("asr_status", "pending"),
                asr_error=row.get("asr_error"), status=c.get("status", "unclassified") if row.get("asr_status") == "complete" else "unclassified",
                wav=row["tts"]["wav_path"]))
    lines = [f"# TTS audit ({metadata['options']['language']}): {expected_total} cases", "",
        "Automatic detector results; these are not independently confirmed listening labels.", "",
        "Seed: " + str(metadata["options"]["seed"]) + "; voice: " + metadata["options"]["voice"], "",
        "Noise suspected: " + str(count["noise_suspected"]) + f"/{expected_total}",
        "Long silence (>1s): " + str(count["long_silence"]) + f"/{expected_total}; silence only: " + str(count["silence_only"]),
        "No anomaly found with ASR word times: " + str(count["no_anomaly_found"]),
        "Unclassified: " + str(count["unclassified"]) + "; synthesis errors: " + str(count["synthesis_errors"]) + "; ASR errors: " + str(count["asr_errors"]), "",
        "Total wall time this invocation: " + format(elapsed, ".3f") + " s",
        "Recorded synthesis: " + format(timing["synthesis_wall_s"], ".3f") + " s; ASR model load: " + format(model_load_s, ".3f")
        + " s; recorded ASR inference: " + format(timing["asr_inference_s"], ".3f") + " s", "",
        "| Case | Text | Audio s | Last word s | Noise? | Silence? | ASR status | WAV |",
        "|---|---|---:|---:|---|---|---|---|"]
    for row in rows:
        e, c = row["entry"], row.get("check") or {}
        last = (c.get("asr_evidence") or {}).get("last_word_end_s")
        path = Path(row["tts"]["wav_path"]).as_posix()
        lines.append("| " + " | ".join([e["id"], e["text"].replace("|", "\\|"),
            format(row["tts"]["audio_s"], ".2f"), format(last, ".2f") if last is not None else "?",
            str(c.get("strange_noise_suspected", "?")), str(c.get("long_silence_suspected", "?")), row.get("asr_status", "pending"),
            "[audio](" + path + ")"]) + " |")
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result

def main():
    began = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path(r"D:\Projekte\miep-codex\config.toml"))
    p.add_argument("--corpus", type=Path, default=CORPUS)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--voice")
    p.add_argument("--language", choices=("es", "en", "pt", "de"), default="es", help="Language for both Qwen TTS and offline ASR; must match the corpus")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--resume", action="store_true", help="Reuse completed WAVs/results; never resynthesize failed/completed cases")
    p.add_argument("--validate-corpus", action="store_true")
    args = p.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8-sig"))
    if corpus.get("language") != args.language:
        raise ValueError("Corpus language must match --language")
    entries = validate_corpus(corpus)
    expected_total = len(entries)
    if args.validate_corpus:
        short_count = sum(entry["class"] == "short" for entry in entries)
        longer_count = sum(entry["class"] == "longer" for entry in entries)
        print(json.dumps({"total":expected_total,"short":short_count,"longer":longer_count,"seed":args.seed,"language":args.language,"corpus_sha256":digest(args.corpus)}))
        return 0
    if not args.output_dir or args.timeout <= 0:
        p.error("--output-dir and positive --timeout are required")
    fw._install_torch_blocker()
    import requests
    import tts_quality_check as qc
    from dotenv import dotenv_values
    cfg = tomllib.loads(args.config.read_text(encoding="utf-8-sig"))["tts"]
    endpoint = cfg["endpoint"].rstrip("/")
    if urlsplit(endpoint).hostname != "192.168.178.22":
        raise ValueError("This benchmark is scoped to the user-owned LAN Qwen server")
    values = dotenv_values(args.config.parent / ".env")
    key = resolve_key(cfg, values)
    headers = {"Authorization": "Bearer " + key}
    response = requests.get(endpoint + "/v1/capabilities", headers=headers, timeout=15)
    response.raise_for_status()
    capabilities = response.json()
    if capabilities["engine"]["device"] != "cpu" or not capabilities["engine"]["name"].startswith("qwen"):
        raise ValueError("Expected the Qwen CPU backend")
    audio = capabilities["audio"]
    rate = audio["sample_rate_hz"]
    if audio["channels"] != 1 or audio["bits_per_sample"] != 16:
        raise ValueError("Expected mono PCM16")
    options = {"endpoint":endpoint,"voice":args.voice or cfg["voice"],"language":args.language,"response_format":"pcm",
        "seed":args.seed,"max_new_tokens":cfg.get("max_new_tokens",256),"temperature":cfg.get("temperature",.9),
        "alignment":{"source":"auto","mode":"live","level":"word"}}
    identity = {"script_sha256":digest(__file__), "server_identity_sha256":hashlib.sha256(json.dumps({k:capabilities[k] for k in ("engine","model","audio")},sort_keys=True).encode()).hexdigest(), "options":options,"corpus_sha256":digest(args.corpus),"detector_sha256":digest(qc.__file__),
        "asr_model_sha256":fw.MODEL_SHA256,"asr_params":{"language":args.language,"word_timestamps":True,"vad_filter":False,
        "condition_on_previous_text":False,"beam_size":3,"temperature":0,"cpu_threads":1,"num_workers":1,"compute_type":"float16"}}
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    directory = args.output_dir.resolve()
    metadata = {"created_utc":datetime.now(timezone.utc).isoformat(),"fingerprint":fingerprint,**identity,
        "tts_capabilities":capabilities,"script_sha256":digest(__file__),"asr_helper_sha256":digest(fw.__file__),
        "reproduction_note":"Fixed order and per-request seed. Native model/reference/build/settings still affect exact PCM."}
    if directory.exists():
        if not args.resume:
            raise ValueError("Output directory exists; choose a new directory or explicit --resume")
        previous = json.loads((directory / "run.json").read_text(encoding="utf-8-sig"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Resume settings/source identity differs")
        metadata = previous
    else:
        directory.mkdir(parents=True)
        for name in ("wav","streams","rows","asr"):
            (directory / name).mkdir()
        write_json(directory / "run.json", metadata)
        write_json(directory / "corpus.json", corpus)
    rows = []
    model_load_s = 0.0
    for number, entry in enumerate(entries, 1):
        checkpoint = directory / "rows" / (entry["id"] + ".json")
        if checkpoint.exists():
            row = json.loads(checkpoint.read_text(encoding="utf-8-sig"))
            if row["entry"] != entry:
                raise ValueError("Checkpoint corpus entry differs: " + entry["id"])
            if row["tts"]["status"] == "complete" and digest(row["tts"]["wav_path"]) != row["tts"]["sha256"]:
                raise ValueError("Recorded WAV changed: " + entry["id"])
        else:
            tts = asyncio.run(synthesize(entry, options, key, rate, args.timeout, directory))
            row = {"entry":entry,"tts":tts,"check":qc.check_wav(Path(tts["wav_path"])) if tts["status"]=="complete" else None}
            write_json(checkpoint,row)
        rows.append(row)
        overview(directory,metadata,rows,time.perf_counter()-began,expected_total=expected_total)
        print(json.dumps({"phase":"tts","case":number,"of":expected_total,"text":entry["text"],"status":row["tts"]["status"],
            "audio_s":round(row["tts"]["audio_s"],3),"wall_s":round(row["tts"]["wall_s"],3)},ensure_ascii=True),flush=True)
    pending = [r for r in rows if r["tts"]["status"]=="complete" and r.get("asr_status") is None]
    if pending:
        if fw._free_gpu_mib() < fw.MIN_FREE_MIB:
            raise RuntimeError("Insufficient free GPU memory for offline large-v2")
        files = {n:fw._model_file(fw.DEFAULT_MODEL_PATH,n) for n in ("config.json","tokenizer.json","vocabulary.txt","model.bin")}
        if files["model.bin"]["sha256"] != fw.MODEL_SHA256:
            raise ValueError("Model weights differ from verified large-v2")
        from faster_whisper import WhisperModel
        load_start = time.perf_counter()
        model = WhisperModel(str(fw.DEFAULT_MODEL_PATH),device="cuda",compute_type="float16",cpu_threads=1,num_workers=1,local_files_only=True)
        model_load_s = time.perf_counter()-load_start
        print(json.dumps({"phase":"asr_model_loaded","wall_s":model_load_s}),flush=True)
        for number,row in enumerate(rows,1):
            if row not in pending:
                continue
            started = time.perf_counter()
            try:
                segments,info=model.transcribe(row["tts"]["wav_path"],language=args.language,word_timestamps=True,
                    vad_filter=False,condition_on_previous_text=False,beam_size=3,temperature=0)
                normalized=[fw._segment(s) for s in segments]
                row["asr_wall_s"]=time.perf_counter()-started
                asr={"source":fw._read_wav(Path(row["tts"]["wav_path"])),
                    "model":{"name":"large-v2","device":"cuda","compute_type":"float16","model_files":files},
                    "params":identity["asr_params"],"timing":{"inference_wall_s":row["asr_wall_s"]},
                    "transcription":{"text":"".join(s["text"] for s in normalized).strip(),"segments":normalized}}
                write_json(directory/"asr"/(row["entry"]["id"]+".json"),asr)
                row["check"]=qc.check_wav(Path(row["tts"]["wav_path"]),asr)
                row["asr_status"]="complete"
            except Exception as exc:
                row["asr_status"]="error"
                row["asr_error"]=type(exc).__name__
                row["asr_wall_s"]=time.perf_counter()-started
            write_json(directory/"rows"/(row["entry"]["id"]+".json"),row)
            overview(directory,metadata,rows,time.perf_counter()-began,model_load_s,expected_total)
            print(json.dumps({"phase":"asr","case":number,"status":row["asr_status"],
                "noise":row["check"]["strange_noise_suspected"],"wall_s":row["asr_wall_s"]}),flush=True)
        del model
    result=overview(directory,metadata,rows,time.perf_counter()-began,model_load_s,expected_total)
    print(json.dumps({"counts":result["counts"],"timing":result["timing"],"output":str(directory)}),flush=True)
    return 0 if result["status"]=="complete" and result["counts"]["unclassified"]==0 else 2

if __name__ == "__main__":
    raise SystemExit(main())
