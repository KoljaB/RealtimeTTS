"""One-clip offline faster-whisper timestamp annotator.

Run this CLI with python -B -S. The process blocks optional Torch imports
because CTranslate2's converter modules otherwise preload Torch's separate
libiomp5md.dll. The blocker is process-local and does not modify packages.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as package_metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave


DEFAULT_MODEL_PATH = Path(
    r"D:\hf_cache\hub\models--Systran--faster-whisper-large-v2"
    r"\snapshots\f0fe81560cb8b68660e564f55dd99207059c092e"
)
MIN_FREE_MIB = 5000
MODEL_SHA256 = "bf2a9746382e1aa7ffff6b3a0d137ed9edbd9670c3b87e5d35f5e85e70d0333a"
SITE_PACKAGES = Path(r"C:\Python\Python312\Lib\site-packages")


class AnnotationError(RuntimeError):
    """A bounded local annotation failure safe to report to the caller."""


class _BlockTorchImports:
    """Keep optional Torch from loading a second OpenMP runtime."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError(
                "Torch is intentionally blocked for this faster-whisper process"
            )
        return None


def _install_torch_blocker() -> None:
    if not sys.flags.no_site:
        raise AnnotationError("Use C:\\Python\\Python312\\python.exe -B -S for this isolated CLI")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    if str(SITE_PACKAGES) not in sys.path:
        sys.path.insert(0, str(SITE_PACKAGES))
    sys.meta_path.insert(0, _BlockTorchImports())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_wav(path: Path) -> dict:
    if not path.is_file():
        raise AnnotationError(f"WAV file not found: {path}")
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AnnotationError("Expected a RIFF/WAVE file")
    if int.from_bytes(data[4:8], "little") + 8 != len(data):
        raise AnnotationError("WAV size does not match the RIFF declaration")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            frames = wav.getnframes()
            compression = wav.getcomptype()
            payload = wav.readframes(frames)
    except (EOFError, OSError, wave.Error) as exc:
        raise AnnotationError(f"Malformed WAV: {type(exc).__name__}") from exc
    if compression != "NONE":
        raise AnnotationError(f"Compressed WAV is unsupported: {compression}")
    if rate < 8000 or channels < 1 or frames < 1 or width not in {1, 2, 3, 4}:
        raise AnnotationError("Unsupported WAV rate, channels, width, or empty audio")
    if len(payload) != frames * channels * width:
        raise AnnotationError("WAV PCM payload is truncated")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(data),
        "bytes": len(data),
        "sample_rate": rate,
        "channels": channels,
        "sample_width_bytes": width,
        "frames": frames,
        "duration_s": frames / rate,
        "compression": compression,
    }


def _free_gpu_mib() -> int:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AnnotationError(f"nvidia-smi preflight failed: {type(exc).__name__}") from exc
    values = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                values.append(int(float(line.split(",", 1)[0].strip())))
            except ValueError as exc:
                raise AnnotationError("nvidia-smi returned an invalid free-memory value") from exc
    if not values:
        raise AnnotationError("nvidia-smi returned no GPU")
    return values[0]


def _model_file(path: Path, name: str) -> dict:
    link = path / name
    try:
        resolved = link.resolve(strict=True)
    except OSError as exc:
        raise AnnotationError(f"Missing model file: {link}") from exc
    return {
        "path": str(link),
        "resolved_path": str(resolved),
        "exists": link.is_file(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
        "symlink": link.is_symlink(),
        "link_target": os.readlink(link) if link.is_symlink() else None,
    }


def _value(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _word(obj) -> dict:
    return {
        "word": _value(obj, "word"),
        "start": _value(obj, "start"),
        "end": _value(obj, "end"),
        "probability": _value(obj, "probability"),
    }


def _segment(obj) -> dict:
    return {
        "id": _value(obj, "id"),
        "start": _value(obj, "start"),
        "end": _value(obj, "end"),
        "text": _value(obj, "text"),
        "temperature": _value(obj, "temperature"),
        "avg_logprob": _value(obj, "avg_logprob"),
        "compression_ratio": _value(obj, "compression_ratio"),
        "no_speech_prob": _value(obj, "no_speech_prob"),
        "words": [_word(word) for word in (_value(obj, "words") or [])],
    }


def _write_new_json(path: Path, value: dict) -> None:
    if path.exists():
        raise AnnotationError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        if path.exists():
            raise AnnotationError(f"Refusing to overwrite existing output: {path}")
        os.rename(temporary, path)  # On Windows, fail if another writer created the output.
    finally:
        if temporary.exists():
            temporary.unlink()


def annotate(wav_path: Path, output: Path, model_path: Path, language: str | None) -> dict:
    if output.exists():
        raise AnnotationError(f"Refusing to overwrite existing output: {output}")
    free_mib = _free_gpu_mib()
    if free_mib < MIN_FREE_MIB:
        raise AnnotationError(
            f"GPU free memory {free_mib} MiB is below the {MIN_FREE_MIB} MiB gate"
        )
    source = _read_wav(wav_path)
    model_path = model_path.resolve(strict=True)
    if not model_path.is_dir():
        raise AnnotationError(f"Model path is not a directory: {model_path}")
    print("MODEL_HASH_START", flush=True)
    model_files = {
        name: _model_file(model_path, name)
        for name in ("config.json", "tokenizer.json", "vocabulary.txt", "model.bin")
    }
    if model_files["model.bin"]["sha256"] != MODEL_SHA256:
        raise AnnotationError("Model weights do not match the verified large-v2 snapshot")
    print(
        "MODEL_HASH_DONE model_sha256="
        + model_files["model.bin"]["sha256"],
        flush=True,
    )
    params = {
        "language": language,
        "word_timestamps": True,
        "vad_filter": False,
        "condition_on_previous_text": False,
        "beam_size": 3,
        "temperature": 0.0,
        "cpu_threads": 1,
        "num_workers": 1,
        "local_files_only": True,
    }
    from faster_whisper import WhisperModel

    load_wall_start = time.perf_counter_ns()
    load_cpu_start = time.process_time_ns()
    model = WhisperModel(
        str(model_path),
        device="cuda",
        compute_type="float16",
        cpu_threads=1,
        num_workers=1,
        local_files_only=True,
    )
    load_timing = {
        "wall_ms": (time.perf_counter_ns() - load_wall_start) / 1e6,
        "process_cpu_ms": (time.process_time_ns() - load_cpu_start) / 1e6,
    }
    print(f"MODEL_LOADED wall_ms={load_timing['wall_ms']:.1f}", flush=True)
    inference_wall_start = time.perf_counter_ns()
    inference_cpu_start = time.process_time_ns()
    segments, info = model.transcribe(
        str(wav_path),
        language=language,
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
        beam_size=3,
        temperature=0.0,
    )
    normalized_segments = [_segment(segment) for segment in segments]
    inference_timing = {
        "wall_ms": (time.perf_counter_ns() - inference_wall_start) / 1e6,
        "process_cpu_ms": (time.process_time_ns() - inference_cpu_start) / 1e6,
    }
    result = {
        "schema_version": 1,
        "status": "complete",
        "source": source,
        "model": {
            "name": "large-v2",
            "backend": "faster-whisper",
            "device": "cuda",
            "compute_type": "float16",
            "snapshot": str(model_path),
            "model_files": model_files,
        },
        "params": params,
        "preflight": {
            "gpu_free_mib_before_load": free_mib,
            "minimum_gpu_free_mib": MIN_FREE_MIB,
            "torch_import_blocked": True,
        },
        "timing": {"load": load_timing, "inference": inference_timing},
        "transcription": {
            "text": "".join(item["text"] or "" for item in normalized_segments).strip(),
            "segments": normalized_segments,
            "info": {
                key: _value(info, key)
                for key in ("language", "language_probability", "duration", "duration_after_vad")
                if _value(info, key) is not None
            },
        },
        "runtime": {
            "python": sys.executable,
            "python_version": sys.version,
            "faster_whisper_version": package_metadata.version("faster-whisper"),
            "ctranslate2_version": package_metadata.version("ctranslate2"),
        },
        "note": (
            "Torch optional imports are blocked only in this process to avoid "
            "a duplicate OpenMP DLL; no global package state is changed."
        ),
    }
    _write_new_json(output, result)
    del model
    gc.collect()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--language", default=None)
    args = parser.parse_args(argv)
    try:
        _install_torch_blocker()
        result = annotate(args.wav.resolve(), args.output.resolve(), args.model_path, args.language)
    except (AnnotationError, OSError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.resolve()),
        "source_sha256": result["source"]["sha256"],
        "word_count": sum(len(item["words"]) for item in result["transcription"]["segments"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
