#!/usr/bin/env python3
"""Compare two Qwen-compatible HTTP speech servers.

The comparator deliberately measures the bytes received by the HTTP client,
not a local playback device.  Both endpoints receive the same serialized JSON
request and requests are interleaved so that a slow machine-wide change is less
likely to bias one side.  The report contains timing, PCM-contract, digest,
and two-client serialization data, but never stores audio or bearer tokens.

Example (the report must be outside this checkout)::

    python tools/benchmark_qwen_servers.py \
        --baseline-url http://127.0.0.1:18084 \
        --candidate-url http://127.0.0.1:18086 \
        --candidate-key-env QWEN_CANDIDATE_KEY \
        --output D:\\Temp\\User\\qwen-server-comparison.json

The default is 30 measured requests per endpoint plus one unmeasured warmup.
Use ``--no-fail`` when collecting a diagnostic report despite a failed
acceptance decision.
"""

from __future__ import annotations

import argparse
import array
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit


DEFAULT_TEXT = "That was a close one, but we made it through."
DEFAULT_VOICE = "mira_v5_spark"
DEFAULT_SAMPLE_RATE = 24_000
DEFAULT_AUDIBLE_THRESHOLD = 80
DEFAULT_READ_BYTES = 8192
DEFAULT_TIMEOUT_SECONDS = 180.0


class BenchmarkError(RuntimeError):
    """A configuration or benchmark orchestration error."""


@dataclass(frozen=True)
class Endpoint:
    name: str
    display_url: str
    scheme: str
    host: str
    port: int
    path: str
    api_key: Optional[str]


@dataclass(frozen=True)
class RequestResult:
    row: dict[str, Any]
    started_ns: int
    ended_ns: int


def _percentile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric_summary(rows: list[dict[str, Any]], field: str) -> Optional[dict[str, float]]:
    values = [
        float(row[field])
        for row in rows
        if row.get("http_status") == 200 and _finite(row.get(field))
    ]
    if not values:
        return None
    return {
        "count": float(len(values)),
        "p50": float(_percentile(values, 0.50)),
        "median": float(_percentile(values, 0.50)),
        "p95": float(_percentile(values, 0.95)),
        "mean": float(sum(values) / len(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _safe_error(exc: BaseException) -> str:
    """Return a bounded, non-body error description.

    In particular, this function is never passed a response body.  An error
    body could contain user text or implementation details that do not belong
    in a benchmark report.
    """

    message = str(exc).replace("\r", " ").replace("\n", " ").strip()
    if not message:
        message = exc.__class__.__name__
    return message[:240]


def _safe_display_url(value: str) -> tuple[str, str, str, int, str]:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise BenchmarkError(f"invalid endpoint URL: {_safe_error(exc)}") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise BenchmarkError("endpoint URL scheme must be http or https")
    if not host:
        raise BenchmarkError("endpoint URL must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise BenchmarkError("endpoint URL must not contain userinfo")
    if parsed.query or parsed.fragment:
        raise BenchmarkError("endpoint URL must not contain a query or fragment")
    scheme = parsed.scheme.lower()
    if port is None:
        port = 443 if scheme == "https" else 80
    if not 1 <= port <= 65535:
        raise BenchmarkError("endpoint URL port must be between 1 and 65535")
    base_path = parsed.path.rstrip("/")
    path = f"{base_path}/v1/audio/speech" if base_path else "/v1/audio/speech"
    # The original URL is safe after rejecting userinfo, query, and fragment.
    display = urlunsplit((scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
    return display, scheme, host, port, path


def _endpoint(name: str, value: str, api_key: Optional[str]) -> Endpoint:
    display, scheme, host, port, path = _safe_display_url(value.strip())
    return Endpoint(name, display, scheme, host, port, path, api_key or None)


def _resolve_key(explicit: str, env_name: str) -> Optional[str]:
    if explicit and env_name:
        raise BenchmarkError("an endpoint key and its key environment variable are mutually exclusive")
    if explicit:
        return explicit
    if env_name:
        value = os.environ.get(env_name)
        if value is None:
            raise BenchmarkError(f"key environment variable is not set: {env_name}")
        return value.strip() or None
    return None


def _request_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input": args.text,
        "voice": args.voice,
        "response_format": "pcm",
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.language is not None:
        payload["language"] = args.language
    if args.instructions is not None:
        payload["instructions"] = args.instructions
    return payload


def _read_response_chunk(response: http.client.HTTPResponse, read_bytes: int) -> bytes:
    reader = getattr(response, "read1", None)
    if reader is not None:
        return bytes(reader(read_bytes))
    return bytes(response.read(read_bytes))


def _sample_stats(data: bytes, carry: bytes, threshold: int) -> tuple[bytes, int, int, Optional[int]]:
    combined = carry + data
    complete_length = len(combined) - (len(combined) % 2)
    next_carry = combined[complete_length:]
    if complete_length == 0:
        return next_carry, 0, 0, None
    values = array.array("h")
    values.frombytes(combined[:complete_length])
    if sys.byteorder != "little":
        values.byteswap()
    peak = max((abs(int(value)) for value in values), default=0)
    first_audible: Optional[int] = None
    if threshold > 0:
        for index, value in enumerate(values):
            if abs(int(value)) >= threshold:
                first_audible = index
                break
    return next_carry, len(values), peak, first_audible


def _header_int(response: http.client.HTTPResponse, name: str) -> Optional[int]:
    value = response.getheader(name)
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _request_once(
    endpoint: Endpoint,
    body: bytes,
    *,
    index: int,
    phase: str,
    sample_rate: int,
    audible_threshold: int,
    read_bytes: int,
    timeout: float,
    start_gate: Optional[threading.Barrier] = None,
) -> RequestResult:
    if start_gate is not None:
        try:
            start_gate.wait(timeout=timeout)
        except threading.BrokenBarrierError as exc:
            now = time.perf_counter_ns()
            return RequestResult(
                {
                    "index": index,
                    "phase": phase,
                    "endpoint": endpoint.name,
                    "http_status": None,
                    "error": "probe_start_barrier_broken",
                },
                now,
                now,
            )

    started_ns = time.perf_counter_ns()
    ended_ns = started_ns
    connection: Optional[http.client.HTTPConnection] = None
    row: dict[str, Any] = {
        "index": index,
        "phase": phase,
        "endpoint": endpoint.name,
        "http_status": None,
        "first_body_ms": None,
        "first_audible_ms": None,
        "wall_ms": None,
        "audio_duration_s": None,
        "rtf": None,
        "bytes": 0,
        "pcm_valid": False,
        "pcm_sha256": None,
        "sample_rate_hz": None,
        "sample_rate_status": "unavailable",
        "media_type": None,
        "media_type_valid": False,
        "audio_contract_status": "unavailable",
        "audio_contract_valid": False,
        "audio_contract": {
            "encoding": None,
            "channels": None,
            "bits_per_sample": None,
        },
    }
    try:
        connection_class = (
            http.client.HTTPSConnection
            if endpoint.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_class(endpoint.host, endpoint.port, timeout=timeout)
        headers = {
            "Accept": "audio/pcm",
            "Content-Type": "application/json",
            "Connection": "close",
        }
        if endpoint.api_key:
            headers["Authorization"] = f"Bearer {endpoint.api_key}"
        connection.request("POST", endpoint.path, body=body, headers=headers)
        response = connection.getresponse()
        row["http_status"] = int(response.status)
        if response.status != 200:
            # Drain only a small bounded amount so the connection can close.
            try:
                response.read(4096)
            except (OSError, http.client.HTTPException):
                pass
            row["error"] = f"http_status_{int(response.status)}"
            ended_ns = time.perf_counter_ns()
            row["wall_ms"] = (ended_ns - started_ns) / 1_000_000.0
            return RequestResult(row, started_ns, ended_ns)

        content_type = response.getheader("Content-Type") or ""
        media_type = str(content_type).split(";", 1)[0].strip().lower() or None
        row["media_type"] = media_type
        row["media_type_valid"] = media_type == "audio/pcm"

        raw_rate = response.getheader("X-Audio-Sample-Rate")
        header_rate = _header_int(response, "X-Audio-Sample-Rate")
        if raw_rate is None:
            row["sample_rate_hz"] = sample_rate
            row["sample_rate_status"] = "assumed_configured_rate"
        elif header_rate is not None and header_rate > 0 and header_rate == sample_rate:
            row["sample_rate_hz"] = header_rate
            row["sample_rate_status"] = "verified"
        else:
            row["sample_rate_hz"] = sample_rate
            row["sample_rate_status"] = "mismatch"

        header_channels = _header_int(response, "X-Audio-Channels")
        header_bits = _header_int(response, "X-Audio-Bits-Per-Sample")
        header_encoding = response.getheader("X-Audio-Encoding")
        encoding = str(header_encoding).strip().lower() if header_encoding is not None else None
        row["audio_contract"] = {
            "encoding": encoding,
            "channels": header_channels,
            "bits_per_sample": header_bits,
        }
        contract_mismatch = (
            (header_encoding is not None and encoding not in {"s16le", "pcm_s16le"})
            or (response.getheader("X-Audio-Channels") is not None and header_channels != 1)
            or (response.getheader("X-Audio-Bits-Per-Sample") is not None and header_bits != 16)
        )
        any_contract_header = any(
            value is not None
            for value in (header_encoding, response.getheader("X-Audio-Channels"), response.getheader("X-Audio-Bits-Per-Sample"))
        )
        row["audio_contract_status"] = (
            "mismatch" if contract_mismatch else "verified" if any_contract_header else "assumed_configured_pcm16_mono"
        )
        row["audio_contract_valid"] = not contract_mismatch
        body_hash = hashlib.sha256()
        total_bytes = 0
        sample_count = 0
        peak = 0
        carry = b""
        playback_cursor_ns: Optional[int] = None
        first_body_ns: Optional[int] = None
        first_audible_ns: Optional[int] = None
        while True:
            chunk = _read_response_chunk(response, read_bytes)
            if not chunk:
                break
            arrival_ns = time.perf_counter_ns()
            if first_body_ns is None:
                first_body_ns = arrival_ns
            body_hash.update(chunk)
            total_bytes += len(chunk)
            carry, chunk_samples, chunk_peak, audible_index = _sample_stats(
                chunk, carry, audible_threshold
            )
            chunk_start_ns = max(arrival_ns, playback_cursor_ns or arrival_ns)
            if audible_index is not None and first_audible_ns is None:
                first_audible_ns = chunk_start_ns + int(
                    audible_index * 1_000_000_000 / sample_rate
                )
            if chunk_samples:
                playback_cursor_ns = chunk_start_ns + int(
                    chunk_samples * 1_000_000_000 / sample_rate
                )
            sample_count += chunk_samples
            peak = max(peak, chunk_peak)

        ended_ns = time.perf_counter_ns()
        row["bytes"] = total_bytes
        row["pcm_sha256"] = body_hash.hexdigest()
        row["pcm_valid"] = bool(total_bytes and not carry)
        row["sample_count"] = sample_count
        row["peak_abs"] = peak
        row["sample_rate_valid"] = row["sample_rate_status"] != "mismatch"
        row["media_type_valid"] = bool(row["media_type_valid"])
        row["audio_duration_s"] = sample_count / float(row["sample_rate_hz"])
        row["first_body_ms"] = (
            (first_body_ns - started_ns) / 1_000_000.0 if first_body_ns is not None else None
        )
        row["first_audible_ms"] = (
            (first_audible_ns - started_ns) / 1_000_000.0
            if first_audible_ns is not None
            else None
        )
        row["wall_ms"] = (ended_ns - started_ns) / 1_000_000.0
        row["rtf"] = (
            (ended_ns - started_ns) / 1_000_000_000.0 / float(row["audio_duration_s"])
            if row["audio_duration_s"]
            else None
        )
        if not row["pcm_valid"]:
            row["error"] = "invalid_pcm_length_or_empty_audio"
        elif not row["sample_rate_valid"]:
            row["error"] = "sample_rate_mismatch"
        elif not row["media_type_valid"]:
            row["error"] = "unexpected_content_type"
        elif not row["audio_contract_valid"]:
            row["error"] = "audio_contract_mismatch"
        return RequestResult(row, started_ns, ended_ns)
    except (OSError, TimeoutError, http.client.HTTPException) as exc:
        ended_ns = time.perf_counter_ns()
        row["wall_ms"] = (ended_ns - started_ns) / 1_000_000.0
        row["error"] = f"transport_error: {_safe_error(exc)}"
        return RequestResult(row, started_ns, ended_ns)
    finally:
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("http_status") == 200]
    valid_pcm = [row for row in successful if row.get("pcm_valid")]
    valid_rate = [row for row in successful if row.get("sample_rate_valid")]
    audible = [row for row in successful if row.get("first_audible_ms") is not None]
    hashes = [str(row["pcm_sha256"]) for row in valid_pcm if row.get("pcm_sha256")]
    metrics: dict[str, Any] = {}
    for field in (
        "first_body_ms",
        "first_audible_ms",
        "wall_ms",
        "audio_duration_s",
        "rtf",
    ):
        metric = _metric_summary(rows, field)
        if metric is not None:
            # Counts are integers in the report even though percentile math uses
            # floats internally.
            metric["count"] = int(metric["count"])
            metrics[field] = metric
    return {
        "requested": len(rows),
        "successful_http": len(successful),
        "failed_http": len(rows) - len(successful),
        "pcm_valid": len(valid_pcm),
        "sample_rate_valid": len(valid_rate),
        "media_type_valid": sum(bool(row.get("media_type_valid")) for row in successful),
        "audio_contract_valid": sum(bool(row.get("audio_contract_valid")) for row in successful),
        "audible": len(audible),
        "unique_pcm_hashes": len(set(hashes)),
        "deterministic_pcm": len(set(hashes)) <= 1 if hashes else False,
        "metrics": metrics,
    }


def _metric_value(summary: Mapping[str, Any], field: str, percentile: str) -> Optional[float]:
    value = summary.get("metrics", {}).get(field, {}).get(percentile)
    return float(value) if _finite(value) else None


def _ratio(candidate: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if candidate is None or baseline is None or baseline <= 0:
        return None
    return candidate / baseline


def _run_acceptance(
    args: argparse.Namespace,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    serialization_probe: Mapping[str, Any],
) -> dict[str, Any]:
    criteria: dict[str, bool] = {
        "at_least_30_measured_runs": args.runs >= 30,
        "baseline_http_complete": baseline["successful_http"] == args.runs,
        "candidate_http_complete": candidate["successful_http"] == args.runs,
        "baseline_pcm_valid": baseline["pcm_valid"] == args.runs,
        "candidate_pcm_valid": candidate["pcm_valid"] == args.runs,
        "baseline_sample_rate_valid": baseline["sample_rate_valid"] == args.runs,
        "candidate_sample_rate_valid": candidate["sample_rate_valid"] == args.runs,
        "baseline_media_type_valid": baseline["media_type_valid"] == args.runs,
        "candidate_media_type_valid": candidate["media_type_valid"] == args.runs,
        "baseline_audio_contract_valid": baseline["audio_contract_valid"] == args.runs,
        "candidate_audio_contract_valid": candidate["audio_contract_valid"] == args.runs,
        "candidate_audible_on_every_run": candidate["audible"] == args.runs,
        "baseline_two_client_probe_complete": (
            serialization_probe["baseline"]["successful_http"] == 2
            and serialization_probe["baseline"]["pcm_valid"] == 2
        ),
        "candidate_two_client_probe_complete": (
            serialization_probe["candidate"]["successful_http"] == 2
            and serialization_probe["candidate"]["pcm_valid"] == 2
        ),
    }
    ratios: dict[str, Optional[float]] = {}
    for field in ("first_body_ms", "first_audible_ms", "wall_ms", "rtf"):
        for percentile in ("median", "p95"):
            key = f"{field}_{'p50' if percentile == 'median' else 'p95'}_ratio"
            value = _ratio(
                _metric_value(candidate, field, percentile),
                _metric_value(baseline, field, percentile),
            )
            ratios[key] = value
            criteria[f"candidate_{field}_{'p50' if percentile == 'median' else 'p95'}_within_ratio"] = (
                value is not None and value <= args.max_ratio
            )

    pairs = min(len(baseline_rows), len(candidate_rows))
    valid_pairs = 0
    matching_pairs = 0
    for index in range(pairs):
        baseline_row = baseline_rows[index]
        candidate_row = candidate_rows[index]
        if baseline_row.get("pcm_valid") and candidate_row.get("pcm_valid"):
            valid_pairs += 1
            if baseline_row.get("pcm_sha256") == candidate_row.get("pcm_sha256"):
                matching_pairs += 1
    hash_match_rate = matching_pairs / valid_pairs if valid_pairs else None
    criteria["paired_pcm_hash_match_rate"] = (
        hash_match_rate is not None and hash_match_rate >= args.min_hash_match_rate
    )
    return {
        "passed": all(criteria.values()),
        "max_ratio": args.max_ratio,
        "minimum_hash_match_rate": args.min_hash_match_rate,
        "criteria": criteria,
        "ratios": ratios,
        "paired_pcm_hashes": {
            "valid_pairs": valid_pairs,
            "matching_pairs": matching_pairs,
            "match_rate": hash_match_rate,
        },
    }


def _serialization_probe(
    endpoint: Endpoint,
    body: bytes,
    *,
    sample_rate: int,
    audible_threshold: int,
    read_bytes: int,
    timeout: float,
) -> dict[str, Any]:
    barrier = threading.Barrier(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _request_once,
                endpoint,
                body,
                index=index,
                phase="serialization_probe",
                sample_rate=sample_rate,
                audible_threshold=audible_threshold,
                read_bytes=read_bytes,
                timeout=timeout,
                start_gate=barrier,
            )
            for index in (0, 1)
        ]
        results = [future.result() for future in futures]
    rows = [result.row for result in results]
    starts = [result.started_ns for result in results]
    ends = [result.ended_ns for result in results]
    overall_wall_ms = (max(ends) - min(starts)) / 1_000_000.0 if ends else None
    request_walls = [
        float(row["wall_ms"])
        for row in rows
        if _finite(row.get("wall_ms"))
    ]
    return {
        "requests": rows,
        "successful_http": sum(row.get("http_status") == 200 for row in rows),
        "pcm_valid": sum(bool(row.get("pcm_valid")) for row in rows),
        "overall_wall_ms": overall_wall_ms,
        "sum_request_wall_ms": sum(request_walls) if request_walls else None,
        "serialization_factor": (
            sum(request_walls) / overall_wall_ms
            if request_walls and overall_wall_ms and overall_wall_ms > 0
            else None
        ),
    }


def _interleave_order(index: int) -> tuple[str, str]:
    return ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")


def _run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    baseline_key = _resolve_key(args.baseline_key, args.baseline_key_env)
    candidate_key = _resolve_key(args.candidate_key, args.candidate_key_env)
    baseline = _endpoint("baseline", args.baseline_url, baseline_key)
    candidate = _endpoint("candidate", args.candidate_url, candidate_key)
    endpoints = {"baseline": baseline, "candidate": candidate}
    payload = _request_payload(args)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    warmup_rows: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    for index in range(args.warmups):
        for name in _interleave_order(index):
            result = _request_once(
                endpoints[name],
                body,
                index=index,
                phase="warmup",
                sample_rate=args.sample_rate,
                audible_threshold=args.audible_threshold,
                read_bytes=args.read_bytes,
                timeout=args.timeout,
            )
            warmup_rows[name].append(result.row)

    measured_rows: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    order: list[list[str]] = []
    for index in range(args.runs):
        request_order = list(_interleave_order(index))
        order.append(request_order)
        for name in request_order:
            result = _request_once(
                endpoints[name],
                body,
                index=index,
                phase="measurement",
                sample_rate=args.sample_rate,
                audible_threshold=args.audible_threshold,
                read_bytes=args.read_bytes,
                timeout=args.timeout,
            )
            measured_rows[name].append(result.row)

    probe = {
        name: _serialization_probe(
            endpoints[name],
            body,
            sample_rate=args.sample_rate,
            audible_threshold=args.audible_threshold,
            read_bytes=args.read_bytes,
            timeout=args.timeout,
        )
        for name in ("baseline", "candidate")
    }
    summaries = {
        name: _summary(measured_rows[name]) for name in ("baseline", "candidate")
    }
    acceptance = _run_acceptance(
        args,
        summaries["baseline"],
        summaries["candidate"],
        measured_rows["baseline"],
        measured_rows["candidate"],
        probe,
    )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "benchmark_qwen_servers",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "request": {
            "payload": payload,
            "payload_sha256": hashlib.sha256(body).hexdigest(),
            "sample_rate_hz_expected": args.sample_rate,
            "audible_threshold": args.audible_threshold,
            "read_bytes": args.read_bytes,
        },
        "endpoints": {
            name: {
                "url": endpoint.display_url,
                "speech_path": endpoint.path,
                "bearer_key_configured": endpoint.api_key is not None,
            }
            for name, endpoint in endpoints.items()
        },
        "configuration": {
            "runs_requested": args.runs,
            "warmups_requested": args.warmups,
            "interleaved": True,
            "interleave_order": order,
            "timeout_seconds": args.timeout,
            "max_latency_and_rtf_ratio": args.max_ratio,
            "minimum_hash_match_rate": args.min_hash_match_rate,
        },
        "warmups": warmup_rows,
        "runs": measured_rows,
        "summaries": summaries,
        "serialization_probe": probe,
        "acceptance": acceptance,
    }


def _report_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        path.relative_to(repo_root)
    except ValueError:
        return path
    raise BenchmarkError(
        "--output must be outside the repository; use an explicit path such as "
        "D:\\Temp\\User\\qwen-server-comparison.json"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two Qwen-compatible HTTP PCM speech endpoints"
    )
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument(
        "--baseline-key",
        default="",
        help="Bearer token (prefer --baseline-key-env; never printed or stored)",
    )
    parser.add_argument(
        "--candidate-key",
        default="",
        help="Bearer token (prefer --candidate-key-env; never printed or stored)",
    )
    parser.add_argument("--baseline-key-env", default="")
    parser.add_argument("--candidate-key-env", default="")
    parser.add_argument("--output", required=True, help="JSON report path outside this repository")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--language", default=None)
    parser.add_argument("--instructions", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--audible-threshold", type=int, default=DEFAULT_AUDIBLE_THRESHOLD)
    parser.add_argument("--read-bytes", type=int, default=DEFAULT_READ_BYTES)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=1.15,
        help="Maximum candidate/baseline ratio for each p50/p95 latency and RTF metric",
    )
    parser.add_argument(
        "--min-hash-match-rate",
        type=float,
        default=0.0,
        help=(
            "Minimum paired PCM SHA-256 match rate required for acceptance; "
            "defaults to informational-only because trimming may change bytes"
        ),
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit zero after writing the report",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups must not be negative")
    if args.sample_rate <= 0 or args.audible_threshold < 0 or args.read_bytes <= 0:
        parser.error("sample rate must be positive; threshold must not be negative; read size must be positive")
    if args.timeout <= 0 or args.max_ratio <= 0:
        parser.error("--timeout and --max-ratio must be positive")
    if not 0.0 <= args.min_hash_match_rate <= 1.0:
        parser.error("--min-hash-match-rate must be between 0 and 1")
    if not args.text.strip() or not args.voice.strip():
        parser.error("--text and --voice must not be empty")
    try:
        output = _report_path(args.output)
        report = _run_benchmark(args)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (BenchmarkError, OSError, ValueError) as exc:
        print(f"benchmark error: {_safe_error(exc)}", file=sys.stderr)
        return 2

    acceptance = report["acceptance"]
    print(
        f"report={output} acceptance={'PASS' if acceptance['passed'] else 'FAIL'} "
        f"runs={args.runs}"
    )
    return 0 if acceptance["passed"] or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
