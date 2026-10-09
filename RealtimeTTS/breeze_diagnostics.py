"""Local, failure-only Breeze reproduction records. No synthesis retries."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


def file_identity(path, *, hash_contents=True):
    if not path:
        return None
    source = Path(path).resolve()
    result = {"path": str(source)}
    try:
        stat = source.stat()
        result.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if hash_contents:
            result["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        result["unavailable"] = str(exc)
    return result


def write_failure(directory, request, state, *, worker_state=None, model_path=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    pcm = b"".join(state.chunks if state.response_format == "wav" else state.diagnostic_chunks)
    samples = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype="<i2").astype(np.float64)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    mean_abs = float(np.mean(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
    pcm_file = None
    if pcm:
        pcm_file = record_id + ".wav"
        with wave.open(str(directory / pcm_file), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(pcm)
    engine = state.engine_diagnostics or {}
    voice = engine.get("voice", {})
    reference = file_identity(voice.get("ref_audio"))
    # Preserve the exact reference used by this failure, even if the registry is later replaced.
    if reference and "sha256" in reference:
        suffix = Path(reference["path"]).suffix or ".wav"
        reference_file = "reference-" + reference["sha256"] + suffix
        target = directory / reference_file
        if not target.exists():
            target.write_bytes(Path(reference["path"]).read_bytes())
        reference["saved_file"] = reference_file
    native = None
    if worker_state is not None:
        try:
            native = json.loads(Path(worker_state).read_text(encoding="utf-8-sig"))
            native["binary"] = file_identity(native.get("executable"))
        except (OSError, ValueError) as exc:
            native = {"unavailable": str(exc)}
    from .qwen_server import AUDIBLE_AVERAGE_ABS_THRESHOLD
    record = {
        "schema_version": 1, "record_id": record_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": state.request_id,
        "request": {"input": request.text, "voice": request.voice,
                    "instructions": request.instructions, "language": request.language,
                    "sampling_overrides": request.sampling},
        "error": {"type": state.error_type, "message": state.error_message,
                  "cancelled": state.cancelled.is_set(), "timed_out": state.timed_out},
        "audio": {"received_bytes": state.received_bytes, "audible": state.audible,
                  "saved_bytes": len(pcm), "saved_wav": pcm_file,
                  "sample_rate": 24000, "peak": peak, "mean_abs": mean_abs, "rms": rms,
                  "audibility_mean_abs_threshold": AUDIBLE_AVERAGE_ABS_THRESHOLD,
                  "classification": "no_pcm" if not state.received_bytes else
                      "all_zero" if peak == 0 and pcm else
                      "below_audibility_threshold" if not state.audible else "partial_audio"},
        "engine": engine, "reference": reference, "native_worker": native,
        # Model identity is deliberately cheap; bit-exact replay additionally requires
        # the unchanged GGUF and device/backend. Binary and Python sources are hashed.
        "model_file": file_identity(model_path, hash_contents=False),
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "sources": [file_identity(Path(__file__).with_name(name)) for name in
                                ("qwen_server.py", "breeze_server.py")] +
                               [file_identity(Path(__file__).parent / "engines" / "breeze_engine.py")]},
        "replay_notes": [
            "Replay preceding native blocks in order, preserving native session resets.",
            "Verify preceding PCM hashes; a mismatch means context was not reproduced.",
            "Seed and text alone do not identify the reference cache, voice, directions or decoder context.",
            "Cancelled native controls and GPU/backend nondeterminism can prevent bit-identical replay."
        ],
    }
    output = directory / (record_id + ".json")
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.error("BREEZE_FAILURE request=%s classification=%s record=%s",
                 state.request_id, record["audio"]["classification"], output)
    return output
