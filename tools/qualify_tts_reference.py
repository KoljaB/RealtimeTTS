"""Run a frozen four-seed French TTS reference qualification.

Serial provider synthesis is followed by immediate offline large-v2 word
timestamps. Strict qualification stops on the first noise, long silence, error,
or U result. --diagnostic-full completes acoustic/U findings without qualifying
the reference; infrastructure errors still stop the run.
"""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import asdict
import gc
import hashlib
import json
import shutil
from pathlib import Path
import sys
import time
import tomllib
from urllib.parse import urlsplit
requests = None
dotenv_values = None
qc = None
from benchmark_spanish_tts_50 import fw, overview, resolve_key, synthesize, totals, validate_corpus, write_json

SEEDS = (42, 43, 69, 177)
CASE_COUNT = 100
EXIT_CODES = {"noise_suspected": 20, "long_silence": 21, "synthesis_error": 22, "asr_error": 23, "unclassified": 24, "resume_error": 25}
ASR_PARAMS = {"word_timestamps": True, "vad_filter": False, "condition_on_previous_text": False, "beam_size": 3, "temperature": 0, "cpu_threads": 1, "num_workers": 1, "compute_type": "float16", "no_speech_threshold": None}

class QualificationStop(RuntimeError):
    def __init__(self, reason, message, code=None):
        super().__init__(message)
        self.reason, self.code = reason, code or EXIT_CODES[reason]

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stable(value): return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def json_sha(value): return hashlib.sha256(stable(value).encode()).hexdigest()

def source_metadata(path):
    path = Path(path).resolve()
    meta = dict(fw._read_wav(path)); meta["path"] = str(path)
    if meta.get("sha256") != digest(path): raise ValueError("Reference WAV hash changed")
    if meta.get("channels") != 1 or meta.get("sample_rate") != 24000: raise ValueError("Reference WAV must be mono 24 kHz")
    if meta.get("sample_width_bytes", 2) != 2: raise ValueError("Reference WAV must be PCM16")
    return meta

def provider(config_path, voice, language):
    cfg = tomllib.loads(Path(config_path).read_text(encoding="utf-8-sig"))["tts"]
    endpoint = cfg["endpoint"].rstrip("/")
    if urlsplit(endpoint).hostname != "192.168.178.22": raise ValueError("Provider must be the authorized Qwen LAN endpoint")
    key = resolve_key(cfg, dotenv_values(str(Path(config_path).parent / ".env")))
    if not key: raise ValueError("TTS authentication is not configured")
    headers = {"Authorization": "Bearer " + key}
    response = requests.get(endpoint + "/v1/capabilities", headers=headers, timeout=15); response.raise_for_status()
    caps = response.json(); engine, audio = caps.get("engine", {}), caps.get("audio", {})
    if engine.get("device") != "cpu" or not str(engine.get("name", "")).lower().startswith("qwen"): raise ValueError("Expected Qwen CPU provider")
    if audio.get("sample_rate_hz") != 24000 or audio.get("channels") != 1 or audio.get("bits_per_sample") != 16: raise ValueError("Expected mono PCM16 at 24 kHz")
    options = {"endpoint": endpoint, "voice": voice, "language": language, "response_format": "pcm", "max_new_tokens": cfg.get("max_new_tokens", 256), "temperature": cfg.get("temperature", .9), "alignment": {"source": "auto", "mode": "live", "level": "word"}}
    server_id = hashlib.sha256(json.dumps({k: caps[k] for k in ("engine", "model", "audio")}, sort_keys=True).encode()).hexdigest()
    if server_id != "a71940298a88b95f317d161025aefbcec56cc758e7fcbc10a6fdf598bf6dbb93": raise ValueError("Frozen production CPU profile differs")
    return cfg, key, caps, options, server_id

def profile(args, corpus_sha, reference, detector_sha, runner_sha, helper_sha, model_sha, server_id, options):
    detector_config = asdict(qc.Config()) if hasattr(qc, "Config") else {}
    return {"language": args.language, "diagnostic_full": bool(args.diagnostic_full), "voice": args.voice, "corpus_sha256": corpus_sha, "reference": {"path": reference["path"], "sha256": reference["sha256"], "duration_s": reference.get("duration_s")}, "script_sha256": digest(__file__), "benchmark_helper_sha256": runner_sha, "detector_sha256": detector_sha, "asr_helper_sha256": helper_sha, "asr_model_sha256": model_sha, "server_identity_sha256": server_id, "detector_config": detector_config, "asr_params": dict(ASR_PARAMS, language=args.language), "base_options": options, "seeds": list(SEEDS)}

def run_profile(profile_data, seed):
    return {"profile": profile_data, "options": dict(profile_data["base_options"], seed=seed), "seed": seed, "seed_evidence": "requested_supported_parameter_not_echoed_by_provider"}

def ensure_run(run_dir, run_profile_data, entries, resume):
    run_dir, run_json = Path(run_dir), Path(run_dir) / "run.json"
    if run_dir.exists() and not resume: raise QualificationStop("resume_error", "Output run exists; pass --resume")
    if run_json.exists():
        old = read_json(run_json)
        if old.get("fingerprint") != json_sha(run_profile_data) or old.get("profile") != run_profile_data["profile"] or old.get("options") != run_profile_data["options"]:
            raise QualificationStop("resume_error", "Run fingerprint/source identity differs")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("wav", "streams", "rows", "asr"): (run_dir / name).mkdir()
        write_json(run_json, {**run_profile_data, "fingerprint": json_sha(run_profile_data), "qualification": "Automatic evidence; no human-clean guarantee."})
        write_json(run_dir / "corpus.json", {"entries": entries})

def load_saved(path, entry):
    row = read_json(path)
    if (row.get("entry") or {}).get("id") != entry["id"] or (row.get("entry") or {}).get("text") != entry["text"]: raise QualificationStop("resume_error", "Saved row does not match canonical corpus")
    tts = row.get("tts") or {}
    if tts.get("status") != "complete": raise QualificationStop("resume_error", "Saved synthesis incomplete; refusing regeneration")
    wav = Path(tts.get("wav_path", ""))
    if not wav.is_file() or digest(wav) != tts.get("sha256"): raise QualificationStop("resume_error", "Saved synthesis WAV hash/path invalid")
    return row

def transcribe(model, files, wav, language):
    started = time.perf_counter()
    segments, info = model.transcribe(str(wav), language=language, word_timestamps=True, vad_filter=False, condition_on_previous_text=False, beam_size=3, temperature=0, no_speech_threshold=None)
    normalized = [fw._segment(segment) for segment in segments]
    words = [word for segment in normalized for word in segment.get("words", [])]
    source = dict(fw._read_wav(Path(wav))); text = "".join(segment.get("text", "") for segment in normalized).strip()
    return {"sha256": source["sha256"], "source": source, "model": {"name": "large-v2", "device": "cuda", "compute_type": "float16", "model_files": files}, "params": dict(ASR_PARAMS, language=language), "timing": {"inference_wall_s": time.perf_counter() - started}, "text": text, "words": words, "segments": normalized, "transcription": {"text": text, "segments": normalized}}

def summary(run_dir, metadata, rows, started, model_load_s, stop=None):
    rows = sorted(rows, key=lambda row: row["entry"]["index"])
    charged_load_s = model_load_s if metadata.get("model_load_seed") == metadata["options"]["seed"] else 0.0
    result = overview(run_dir, metadata, rows, time.perf_counter() - started, charged_load_s, expected_total=CASE_COUNT); result["counts"] = totals(rows, expected_total=CASE_COUNT)
    result["timing"]["scope_note"] = "Per-seed active phase time this invocation; reused component records remain in sums; overall job wall is in the root summary."
    if stop: result["stop"] = stop
    write_json(Path(run_dir) / "summary.json", result); return result

def stop_reason(row):
    tts, check = row.get("tts") or {}, row.get("check") or {}
    if tts.get("status") != "complete": return "synthesis_error"
    if row.get("asr_status") == "error": return "asr_error"
    if check.get("strange_noise_suspected"): return "noise_suspected"
    if check.get("long_silence_suspected"): return "long_silence"
    if check.get("asr_status") != "word_timestamps_available": return "unclassified"
    return None

def blocking_reason(row, diagnostic_full):
    reason = stop_reason(row)
    if diagnostic_full and reason in {"noise_suspected", "long_silence", "unclassified"}:
        return None
    return reason

def main():
    job_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(r"D:\Projekte\miep-codex\config.toml"))
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--language", default="fr")
    parser.add_argument("--voice", required=True)
    parser.add_argument("--reference-wav", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,69,177")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostic-full", action="store_true", help="Continue noise, silence, and unavailable-word-timestamp findings through all 400 cases")
    parser.add_argument("--reuse-synthesis-from", type=Path, help="Reuse only source-bound PCM/streams from an explicitly recorded prior study; ASR is always fresh")
    args = parser.parse_args()
    global requests, dotenv_values, qc
    fw._install_torch_blocker()
    import requests as _requests
    from dotenv import dotenv_values as _dotenv_values
    import tts_quality_check as _qc
    requests, dotenv_values, qc = _requests, _dotenv_values, _qc
    if args.language != "fr": raise ValueError("This qualifier is scoped to French; pass --language fr")
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    if seeds != SEEDS: raise ValueError("Seeds must be exactly 42,43,69,177")
    corpus_data = read_json(args.corpus)
    if corpus_data.get("language") != args.language: raise ValueError("Corpus language mismatch")
    entries = validate_corpus(corpus_data)
    if args.timeout <= 0: raise ValueError("Positive case timeout required")
    if digest(qc.__file__) != "b7dcdb394631ef99190b794145228961dbc0bed39e508f762d496d7f0820a322": raise ValueError("Frozen B7 detector differs")
    if len(entries) != CASE_COUNT or len({e["id"] for e in entries}) != CASE_COUNT: raise ValueError("French qualification requires exactly 100 unique corpus entries")
    reference = source_metadata(args.reference_wav)
    cfg, key, capabilities, base_options, server_id = provider(args.config, args.voice, args.language)
    corpus_sha = digest(args.corpus); runner_sha = digest(Path(sys.modules["benchmark_spanish_tts_50"].__file__)); detector_sha = digest(Path(qc.__file__)); helper_sha = digest(Path(fw.__file__))
    study_profile = profile(args, corpus_sha, reference, detector_sha, runner_sha, helper_sha, fw.MODEL_SHA256, server_id, base_options)
    reuse_root = args.reuse_synthesis_from.resolve() if args.reuse_synthesis_from else None
    reuse_fingerprint = None
    if reuse_root:
        source_study = read_json(reuse_root / "study.json")
        source_profile = source_study["profile"]
        for name in ("language", "voice", "corpus_sha256", "reference", "benchmark_helper_sha256", "detector_sha256", "asr_helper_sha256", "asr_model_sha256", "server_identity_sha256", "detector_config", "base_options", "seeds"):
            if source_profile[name] != study_profile[name]: raise ValueError("Synthesis reuse identity differs: " + name)
        reuse_fingerprint = source_study["fingerprint"]
        study_profile["synthesis_reuse_source"] = {"root": str(reuse_root), "fingerprint": reuse_fingerprint, "policy": "PCM and request traces only; discard old ASR/check results"}
    root = args.output_dir.resolve()
    if reuse_root == root: raise ValueError("Fresh output required for ASR profile migration")
    if root.exists() and not args.resume: raise QualificationStop("resume_error", "Output directory exists; pass --resume")
    root.mkdir(parents=True, exist_ok=True); study_path = root / "study.json"
    if study_path.exists():
        old = read_json(study_path)
        if old.get("fingerprint") != json_sha(study_profile): raise QualificationStop("resume_error", "Study fingerprint/source identity differs")
    else:
        write_json(study_path, {"fingerprint": json_sha(study_profile), "profile": study_profile, "capabilities_sha256": json_sha(capabilities), "reference": reference, "status": "running"})
    model = None; model_files = None; model_load_s = 0.0; model_owner_seed = None; study_runs = {}; active_wall = {seed: 0.0 for seed in seeds}
    try:
        for phase_start, phase_end in ((0, 20), (20, CASE_COUNT)):
            for seed in seeds:
                run_dir = root / ("seed-" + str(seed)); run_prof = run_profile(study_profile, seed); ensure_run(run_dir, run_prof, entries, args.resume or phase_start > 0)
                run_json = read_json(run_dir / "run.json"); metadata = {**run_json, "options": run_prof["options"], "requested_seed": seed, "seed_evidence": run_prof["seed_evidence"], "tts_capabilities": {"engine": capabilities.get("engine"), "audio": capabilities.get("audio")}, "model_load_seed": model_owner_seed}
                rows = {}
                for entry in entries[:phase_start]:
                    checkpoint = run_dir / "rows" / (entry["id"] + ".json")
                    if checkpoint.exists(): rows[entry["id"]] = load_saved(checkpoint, entry)
                run_started = time.perf_counter() - active_wall[seed]
                print(json.dumps({"phase": "block_started", "seed": seed, "from_case": phase_start + 1, "through_case": phase_end}), flush=True)
                for entry in entries[phase_start:phase_end]:
                    checkpoint = run_dir / "rows" / (entry["id"] + ".json")
                    if checkpoint.exists(): row = load_saved(checkpoint, entry)
                    else:
                        source_row_path = reuse_root / ("seed-" + str(seed)) / "rows" / (entry["id"] + ".json") if reuse_root else None
                        if source_row_path and source_row_path.exists():
                            saved = load_saved(source_row_path, entry)
                            if saved["entry"] != entry or saved["tts"]["requested_seed"] != seed: raise ValueError("Reused synthesis entry/seed differs")
                            source_wav = Path(saved["tts"]["wav_path"]).resolve()
                            copied_wav = run_dir / "wav" / (entry["id"] + ".wav")
                            if not copied_wav.resolve().is_relative_to(root) or copied_wav.exists(): raise ValueError("Unsafe or existing reuse target")
                            shutil.copyfile(source_wav, copied_wav)
                            if digest(copied_wav) != saved["tts"]["sha256"]: raise ValueError("Reused PCM hash differs")
                            source_stream = reuse_root / ("seed-" + str(seed)) / "streams" / (entry["id"] + ".json")
                            shutil.copyfile(source_stream, run_dir / "streams" / (entry["id"] + ".json"))
                            row = {"entry": entry, "tts": dict(saved["tts"], wav_path=str(copied_wav)), "check": None, "reused_synthesis": {"source_checkpoint": str(source_row_path), "source_wav": str(source_wav), "source_fingerprint": reuse_fingerprint}}
                            row["acoustic_check"] = qc.check_wav(copied_wav)
                            write_json(checkpoint, row)
                    if not checkpoint.exists():
                        if (run_dir / "wav" / (entry["id"] + ".wav")).exists() or (run_dir / "wav" / (entry["id"] + ".partial.wav")).exists(): raise QualificationStop("resume_error", "Orphan WAV preserved; refusing regeneration")
                        tts = asyncio.run(synthesize(entry, run_prof["options"], key, 24000, args.timeout, run_dir)); row = {"entry": entry, "tts": tts, "check": None}
                        if tts.get("status") != "complete":
                            write_json(checkpoint, row); rows[entry["id"]] = row; summary(run_dir, metadata, list(rows.values()), run_started, model_load_s)
                            raise QualificationStop("synthesis_error", "Provider synthesis failed")
                        try: row["acoustic_check"] = qc.check_wav(Path(tts["wav_path"]))
                        except Exception as exc:
                            row["asr_status"], row["asr_error"] = "error", type(exc).__name__; write_json(checkpoint, row)
                            raise QualificationStop("asr_error", "Acoustic checker failed")
                    if row.get("check") is None and row.get("acoustic_check"): row["check"] = row["acoustic_check"]
                    write_json(checkpoint, row)
                    rows[entry["id"]] = row
                    acoustic = row.get("acoustic_check") or {}
                    acoustic_reason = "noise_suspected" if acoustic.get("strange_noise_suspected") else "long_silence" if acoustic.get("long_silence_suspected") else None
                    if acoustic_reason and not args.diagnostic_full:
                        summary(run_dir, metadata, list(rows.values()), run_started, model_load_s, {"reason": acoustic_reason, "case_id": entry["id"]})
                        raise QualificationStop(acoustic_reason, "Acoustic rejection: " + entry["id"])
                    if row.get("asr_status") == "complete" and row.get("check"):
                        reason = blocking_reason(row, args.diagnostic_full)
                        if reason:
                            summary(run_dir, metadata, list(rows.values()), run_started, model_load_s, {"reason": reason, "case_id": entry["id"]})
                            raise QualificationStop(reason, "Saved row already fails qualification")
                        continue
                    asr_path = run_dir / "asr" / (entry["id"] + ".json")
                    try:
                        if asr_path.exists(): asr = read_json(asr_path)
                        else:
                            if model is None:
                                fw._install_torch_blocker(); free = fw._free_gpu_mib()
                                if free < fw.MIN_FREE_MIB: raise RuntimeError("GPU admission below unchanged 5000 MiB gate")
                                model_files = {n: fw._model_file(fw.DEFAULT_MODEL_PATH, n) for n in ("config.json", "tokenizer.json", "vocabulary.txt", "model.bin")}
                                if model_files["model.bin"]["sha256"] != fw.MODEL_SHA256: raise RuntimeError("Cached large-v2 model hash differs")
                                from faster_whisper import WhisperModel
                                loaded = time.perf_counter(); model = WhisperModel(str(fw.DEFAULT_MODEL_PATH), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1, local_files_only=True); model_load_s = time.perf_counter() - loaded
                                model_owner_seed = seed; metadata["model_load_seed"] = seed
                                print(json.dumps({"phase": "asr_model_loaded", "wall_s": model_load_s}), flush=True)
                            asr = transcribe(model, model_files, Path(row["tts"]["wav_path"]), args.language); write_json(asr_path, asr)
                        row["check"] = qc.check_wav(Path(row["tts"]["wav_path"]), asr=asr); row["asr_status"] = "complete"; row["asr_wall_s"] = (asr.get("timing") or {}).get("inference_wall_s")
                    except Exception as exc:
                        row["asr_status"], row["asr_error"] = "error", type(exc).__name__
                        row["asr_error_message"] = str(exc)[:400]
                    write_json(checkpoint, row); rows[entry["id"]] = row
                    current = summary(run_dir, metadata, list(rows.values()), run_started, model_load_s); reason = blocking_reason(row, args.diagnostic_full)
                    print(json.dumps({"phase": "checked", "seed": seed, "case": entry["index"], "noise": row["check"].get("strange_noise_suspected") if row.get("check") else None, "quiet": row["check"].get("long_silence_suspected") if row.get("check") else None, "reason": reason}), flush=True)
                    if reason:
                        current["stop"] = {"reason": reason, "case_id": entry["id"]}; write_json(run_dir / "summary.json", current)
                        raise QualificationStop(reason, "First non-qualifying case: " + entry["id"])
                study_runs[seed] = summary(run_dir, metadata, list(rows.values()), run_started, model_load_s)
                active_wall[seed] = time.perf_counter() - run_started
                print(json.dumps({"phase": "block_complete", "seed": seed, "cases": len(rows), "counts": study_runs[seed]["counts"]}), flush=True)
        all_summaries = [study_runs[s] for s in seeds]; all_counts = {}
        for record in all_summaries:
            for name, value in record["counts"].items():
                if isinstance(value, int): all_counts[name] = all_counts.get(name, 0) + value
        qualified_under_frozen_checks = all_counts.get("synthesis_complete", 0) == all_counts.get("asr_complete", 0) == all_counts.get("no_anomaly_found", 0) == all_counts.get("attempted", 0) == CASE_COUNT * len(SEEDS) and all_counts.get("noise_suspected", 0) == 0 and all_counts.get("long_silence", 0) == 0 and all_counts.get("synthesis_errors", 0) == 0 and all_counts.get("asr_errors", 0) == 0 and all_counts.get("unclassified", 0) == 0
        qualified = qualified_under_frozen_checks and not args.diagnostic_full
        final = {"status": "diagnostic_complete" if args.diagnostic_full else ("qualified_400" if qualified else "complete_with_findings"), "fingerprint": json_sha(study_profile), "profile": study_profile, "counts": all_counts, "qualified_under_frozen_checks": qualified_under_frozen_checks, "synthesis_reused_cases": sum(bool(row.get("reused_synthesis")) for record in all_summaries for row in record["rows"]), "timing": {"overall_wall_s_this_invocation": time.perf_counter() - job_started, "asr_model_load_s": model_load_s, "synthesis_wall_s": sum(study_runs[s]["timing"]["synthesis_wall_s"] for s in seeds), "asr_inference_s": sum(study_runs[s]["timing"]["asr_inference_s"] for s in seeds)}, "runs": [{"seed": s, "summary_path": str(root / ("seed-" + str(s)) / "summary.json"), "counts": study_runs[s]["counts"]} for s in seeds], "qualification": "Automatic acoustic/ASR evidence only; no human-clean guarantee."}
        write_json(root / "summary.json", final); write_json(study_path, {"fingerprint": json_sha(study_profile), "profile": study_profile, "capabilities_sha256": json_sha(capabilities), "reference": reference, "status": final["status"]})
        print(json.dumps({"status": final["status"], "counts": all_counts, "output": str(root)}, ensure_ascii=True)); return 0 if (qualified or args.diagnostic_full) else 2
    except QualificationStop as stop:
        stopped_runs = [{"seed": seed, "summary": read_json(root / ("seed-" + str(seed)) / "summary.json")} for seed in seeds if (root / ("seed-" + str(seed)) / "summary.json").exists()]
        write_json(root / "summary.json", {"status": "stopped", "reason": stop.reason, "exit_code": stop.code, "fingerprint": json_sha(study_profile), "profile": study_profile, "requested_cases": 400, "runs": stopped_runs, "overall_wall_s_this_invocation": time.perf_counter() - job_started})
        write_json(study_path, {"fingerprint": json_sha(study_profile), "profile": study_profile, "status": "stopped", "reason": stop.reason})
        print(json.dumps({"status": "stopped", "reason": stop.reason, "exit_code": stop.code, "message": str(stop)}), flush=True); return stop.code
    finally:
        model = None; gc.collect()

if __name__ == "__main__":
    raise SystemExit(main())
