"""Blind, single-clip Gemini acoustic review helper.

This is an offline annotation tool. It sends one WAV per invocation and never
runs the detector, chooses labels, or edits the corpus manifest.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import wave
from typing import Any

import requests


MODEL_DEFAULT = "gemini-3.5-flash-lite"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
PROMPT_VERSION = "blind-acoustic-review-v1"
MAX_AUDIO_BYTES = 20 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 90.0


class ReviewError(RuntimeError):
    """A local or remote review error safe to report without secrets."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_api_key(env_path: Path) -> str:
    if not env_path.is_file():
        raise ReviewError(f"environment file not found: {env_path}")
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("GEMINI_API_KEY="):
            continue
        value = line.split("=", 1)[1].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            return value
    raise ReviewError("GEMINI_API_KEY is missing from the environment file")


def _read_wav(path: Path) -> tuple[bytes, dict[str, Any]]:
    if not path.is_file():
        raise ReviewError(f"WAV file not found: {path}")
    byte_count = path.stat().st_size
    if byte_count > MAX_AUDIO_BYTES:
        raise ReviewError(
            f"WAV exceeds the inline audio limit ({byte_count} > {MAX_AUDIO_BYTES} bytes)"
        )
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            sample_width = wav.getsampwidth()
            compression = wav.getcomptype()
    except (EOFError, OSError, wave.Error) as exc:
        raise ReviewError(f"invalid WAV header: {type(exc).__name__}") from exc
    if channels <= 0 or sample_rate <= 0 or frames < 0:
        raise ReviewError("WAV header has invalid rate, channel, or frame count")
    if compression != "NONE":
        raise ReviewError(f"compressed WAV is unsupported: {compression}")
    data = path.read_bytes()
    if len(data) != byte_count:
        raise ReviewError("WAV changed while it was being read")
    width_names = {1: "uint8", 2: "int16", 3: "int24", 4: "int32"}
    metadata = {
        "path": str(path),
        "path_sha256": _sha256(data),
        "bytes": len(data),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_format": width_names.get(sample_width, f"pcm_width_{sample_width}"),
        "duration_s": frames / sample_rate,
        "mime_type": "audio/wav",
    }
    return data, metadata


def _review_prompt(clip_id: str, duration_s: float) -> str:
    return f"""You are an independent blind audio reviewer. Listen carefully to the attached WAV clip.
The detector implementation, detector outputs, transcripts, source text, human labels,
and corpus split are unavailable to you. Review only what is audibly present.

Clip metadata: clip_id={clip_id}; duration_s={duration_s:.3f}

Use this definition:
- abnormal means a clearly audible acoustic artifact such as sustained synthesis noise,
  buzzing, a degenerative repeated fragment or loop, unstable gibberish-like sound, or
  another obvious audio-generation failure.
- clean means no clearly audible abnormality of that kind.
- uncertain means the clip is ambiguous or the sound could plausibly be normal speech.
- Silence, pauses, mispronunciation, pronunciation quality, or a cutoff alone are not
  acoustic hallucination.
- Intentional vowels, r sounds, fricatives, whispers, breath, laughter, and other
  expressive speech must be uncertain rather than a confident false alarm.

Return exactly one JSON object, with no Markdown and no extra prose:
{{
  "classification": "clean" | "abnormal" | "uncertain",
  "abnormal_intervals": [
    {{
      "onset_lower_s": number,
      "onset_upper_s": number,
      "end_s": number,
      "type": string,
      "confidence": number,
      "description": string
    }}
  ],
  "other_quality_issues": [string],
  "audible_description": string,
  "cannot_hear": boolean
}}

Use an empty abnormal_intervals list when there is no defensible abnormal interval.
Give approximate interval bounds only when you can hear them; keep every timestamp
inside the clip and do not invent precision. Set cannot_hear true only when the audio
cannot be judged. Confidence is an evidence ranking, not a calibrated probability."""


def _response_text(response_json: dict[str, Any]) -> str:
    texts: list[str] = []
    for candidate in response_json.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            value = part.get("text")
            if isinstance(value, str):
                texts.append(value)
    return "\n".join(texts).strip()


def _json_candidate(text: str) -> Any:
    candidate = text.strip()
    fence = chr(96) * 3
    if candidate.startswith(fence):
        lines = candidate.splitlines()
        if lines and lines[0].strip().startswith(fence):
            lines = lines[1:]
        if lines and lines[-1].strip() == fence:
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(candidate[start:end + 1])


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ReviewError(f"{name} must be finite")
    return number


def _validate_review(value: Any, duration_s: float) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReviewError("model response is not a JSON object")
    classification = value.get("classification")
    if classification not in {"clean", "abnormal", "uncertain"}:
        raise ReviewError("classification is outside the allowed schema")
    intervals = value.get("abnormal_intervals")
    if not isinstance(intervals, list):
        raise ReviewError("abnormal_intervals must be a list")
    normalized: list[dict[str, Any]] = []
    for index, interval in enumerate(intervals):
        if not isinstance(interval, dict):
            raise ReviewError(f"interval {index} is not an object")
        lower = _number(interval.get("onset_lower_s"), f"interval {index} onset_lower_s")
        upper = _number(interval.get("onset_upper_s"), f"interval {index} onset_upper_s")
        end = _number(interval.get("end_s"), f"interval {index} end_s")
        confidence = _number(interval.get("confidence"), f"interval {index} confidence")
        if not 0 <= lower <= upper <= end <= duration_s:
            raise ReviewError(f"interval {index} is outside clip bounds")
        if not 0 <= confidence <= 1:
            raise ReviewError(f"interval {index} confidence is outside [0,1]")
        kind = interval.get("type")
        description = interval.get("description")
        if not isinstance(kind, str) or not isinstance(description, str):
            raise ReviewError(f"interval {index} type/description must be strings")
        normalized.append({
            "onset_lower_s": lower,
            "onset_upper_s": upper,
            "end_s": end,
            "type": kind,
            "confidence": confidence,
            "description": description,
        })
    if classification == "abnormal" and not normalized:
        raise ReviewError("abnormal classification has no interval evidence")
    other = value.get("other_quality_issues")
    audible = value.get("audible_description")
    cannot_hear = value.get("cannot_hear")
    if not isinstance(other, list) or not all(isinstance(item, str) for item in other):
        raise ReviewError("other_quality_issues must be a string list")
    if not isinstance(audible, str) or not isinstance(cannot_hear, bool):
        raise ReviewError("audible_description/cannot_hear have invalid types")
    return {
        "classification": classification,
        "abnormal_intervals": normalized,
        "other_quality_issues": other,
        "audible_description": audible,
        "cannot_hear": cannot_hear,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _matches_existing(path: Path, metadata: dict[str, Any],
                      model: str, prompt_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    previous_audio = previous.get("audio") or {}
    return (
        previous.get("model") == model
        and previous.get("prompt_sha256") == prompt_sha256
        and previous_audio.get("path_sha256") == metadata["path_sha256"]
    )


def _endpoint(model: str) -> str:
    return f"{API_ROOT}/{model}:generateContent"


def review_one(*, wav_path: Path, clip_id: str, output_path: Path,
               env_path: Path, model: str, timeout_s: float,
               force: bool = False) -> dict[str, Any]:
    audio, audio_metadata = _read_wav(wav_path)
    prompt = _review_prompt(clip_id, audio_metadata["duration_s"])
    prompt_sha256 = _sha256(prompt.encode("utf-8"))
    if not force and _matches_existing(output_path, audio_metadata, model, prompt_sha256):
        return {
            "status": "skipped",
            "output": str(output_path),
            "record_id": clip_id,
            "path_sha256": audio_metadata["path_sha256"],
        }
    api_key = _load_api_key(env_path)
    started = _utc_now()
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "error",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "model": model,
        "record_id": clip_id,
        "audio": audio_metadata,
        "request": {
            "method": "POST",
            "endpoint": _endpoint(model),
            "timeout_s": timeout_s,
            "inline_audio": True,
            "api_key_location": "x-goog-api-key",
            "request_audio_bytes": len(audio),
        },
        "started_at_utc": started,
    }
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {
                    "inlineData": {
                        "mimeType": "audio/wav",
                        "data": base64.b64encode(audio).decode("ascii"),
                    }
                },
            ],
        }],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
        },
    }
    try:
        response = requests.post(
            _endpoint(model),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            json=payload,
            timeout=timeout_s,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        result["error"] = f"HTTP request failed: {type(exc).__name__}"
        result["finished_at_utc"] = _utc_now()
        _write_json(output_path, result)
        return result
    result["http_status"] = response.status_code
    try:
        response_json = response.json()
    except ValueError:
        response_json = None
    result["response_raw"] = response_json if response_json is not None else response.text
    result["generation_id"] = (
        response_json.get("responseId")
        if isinstance(response_json, dict) else None
    )
    result["usage"] = (
        response_json.get("usageMetadata")
        if isinstance(response_json, dict) else None
    )
    if response.status_code < 200 or response.status_code >= 300:
        result["error"] = f"HTTP status {response.status_code}"
        result["finished_at_utc"] = _utc_now()
        _write_json(output_path, result)
        return result
    response_text = _response_text(response_json if isinstance(response_json, dict) else {})
    result["response_text"] = response_text
    try:
        parsed = _validate_review(
            _json_candidate(response_text), audio_metadata["duration_s"]
        )
    except (ReviewError, json.JSONDecodeError, TypeError, ValueError) as exc:
        result["status"] = "parse_error"
        result["error"] = f"response schema rejected: {type(exc).__name__}: {exc}"
    else:
        result["status"] = "ok"
        result["review"] = parsed
    result["finished_at_utc"] = _utc_now()
    _write_json(output_path, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(r"D:\Projekte\miep-codex\.env"))
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--force", action="store_true",
                        help="Explicitly replay even when matching output exists.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    try:
        result = review_one(
            wav_path=args.wav.resolve(),
            clip_id=args.clip_id,
            output_path=args.output.resolve(),
            env_path=args.env_file.resolve(),
            model=args.model,
            timeout_s=args.timeout,
            force=args.force,
        )
    except ReviewError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result.get("status"),
        "record_id": result.get("record_id"),
        "output": result.get("output", str(args.output)),
        "http_status": result.get("http_status"),
        "error": result.get("error"),
    }, ensure_ascii=False))
    return 0 if result.get("status") in {"ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
