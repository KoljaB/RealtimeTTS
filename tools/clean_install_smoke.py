#!/usr/bin/env python3
"""Install one built wheel into a disposable venv and verify its boundaries.

The smoke test intentionally installs with ``--no-deps`` and ``--no-index``.
It validates the artifact itself without downloading a model, an optional
engine, or any dependency from a package index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional


def _wheel_from_argument(argument: str) -> Path:
    path = Path(argument).expanduser().resolve()
    if path.is_dir():
        wheels = sorted(path.glob("*.whl"))
        if len(wheels) != 1:
            raise SystemExit(
                f"expected exactly one wheel in {path}, found {len(wheels)}"
            )
        path = wheels[0]
    if path.suffix != ".whl" or not path.is_file():
        raise SystemExit(f"wheel does not exist: {path}")
    return path


def _sdist_from_argument(argument: str) -> Optional[Path]:
    path = Path(argument).expanduser().resolve()
    if not path.is_dir():
        return None
    sdists = sorted(path.glob("*.tar.gz"))
    if len(sdists) != 1:
        raise SystemExit(f"expected exactly one sdist in {path}, found {len(sdists)}")
    return sdists[0]


def _venv_python(environment: Path) -> Path:
    relative = Path("Scripts") / "python.exe" if sys.platform == "win32" else Path("bin") / "python"
    interpreter = environment / relative
    if not interpreter.is_file():
        raise SystemExit(f"virtual-environment interpreter was not created: {interpreter}")
    return interpreter


def _run(command: list[str], *, cwd: Optional[Path] = None) -> None:
    subprocess.run(command, check=True, cwd=cwd)


_CATALOG_PATH = 'RealtimeTTS/voices/catalog.json'
_VOICE_SUFFIXES = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')
_FORBIDDEN_ASSET_SUFFIXES = _VOICE_SUFFIXES + (
    '.spk', '.rvq', '.gguf', '.safetensors', '.ckpt',
    '.pt', '.pth', '.onnx', '.bin', '.model', '.weights',
    '.npy', '.npz', '.tflite', '.mlmodel', '.pb', '.pkl', '.pickle',
)
_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')


def _catalogued_voice_hashes(catalog: bytes) -> dict[str, str]:
    try:
        payload = json.loads(catalog)
    except (TypeError, UnicodeError, ValueError) as error:
        raise SystemExit(f'bundled voice catalog is not valid JSON: {error}') from error
    voices = payload.get('voices') if isinstance(payload, dict) else None
    if not isinstance(voices, list) or not voices:
        raise SystemExit('bundled voice catalog has no voices list')

    expected: dict[str, str] = {}
    for entry in voices:
        if not isinstance(entry, dict):
            raise SystemExit('bundled voice catalog contains a non-object entry')
        file_name = entry.get('file')
        digest = entry.get('sha256')
        if not isinstance(file_name, str) or not file_name:
            raise SystemExit('bundled voice catalog contains an invalid file path')
        relative = PurePosixPath(file_name)
        if (
            chr(92) in file_name
            or relative.is_absolute()
            or relative.as_posix() != file_name
            or any(part in {'', '.', '..'} for part in relative.parts)
            or relative.suffix.lower() not in _VOICE_SUFFIXES
        ):
            raise SystemExit(
                f'bundled voice catalog contains an unsafe voice path: {file_name}'
            )
        archive_name = f'RealtimeTTS/voices/{relative.as_posix()}'
        if archive_name in expected:
            raise SystemExit(f'bundled voice catalog repeats: {file_name}')
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest.lower()):
            raise SystemExit(f'bundled voice catalog has an invalid SHA-256: {file_name}')
        expected[archive_name] = digest.lower()
    return expected


def _validate_catalogued_assets(
    names: set[str],
    expected_hashes: dict[str, str],
    read_bytes,
    artifact: str,
) -> None:
    missing = sorted(set(expected_hashes) - names)
    if missing:
        raise SystemExit(
            f'{artifact} is missing catalogued voice files: ' + ', '.join(missing)
        )
    unexpected = sorted(
        name
        for name in names
        if name.lower().endswith(_FORBIDDEN_ASSET_SUFFIXES)
        and name not in expected_hashes
    )
    if unexpected:
        raise SystemExit(
            f'{artifact} unexpectedly bundles model or voice assets: '
            + ', '.join(unexpected)
        )
    mismatches = []
    for name, expected in sorted(expected_hashes.items()):
        actual = hashlib.sha256(read_bytes(name)).hexdigest()
        if actual != expected:
            mismatches.append(f'{name} (expected {expected}, got {actual})')
    if mismatches:
        raise SystemExit(
            f'{artifact} has catalogued voice hash mismatches: '
            + ', '.join(mismatches)
        )


def _check_wheel_contents(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    required_files = {
        "RealtimeTTS/_version.py",
        "RealtimeTTS/qwen_server.py",
        "RealtimeTTS/qwen_emotions.py",
        "RealtimeTTS/engines/qwen_cpu_engine.py",
        "RealtimeTTS/engines/qwen_engine.py",
        "RealtimeTTS/engines/inflect_engine.py",
        "RealtimeTTS/engines/higgs_engine.py",
        "RealtimeTTS/alignment/__init__.py",
    }
    required_files.update({
        'RealtimeTTS/engines/breeze_engine.py',
        'RealtimeTTS/engines/speechify_engine.py',
        'RealtimeTTS/breeze_server.py',
        'RealtimeTTS/breeze_diagnostics.py',
        'RealtimeTTS/text_exchange.py',
        'RealtimeTTS/alignment/streaming.py',
        'RealtimeTTS/alignment/onnx_ctc.py',
        'RealtimeTTS/alignment/onnx_parakeet.py',
        'RealtimeTTS/alignment/timeline.py',
        'RealtimeTTS/alignment/controller.py',
        'RealtimeTTS/alignment/backends.py',
        'RealtimeTTS/alignment/decoder.py',
        'RealtimeTTS/alignment/server.py',
        'RealtimeTTS/server.py',
        'RealtimeTTS/breeze_worker.py',
        'RealtimeTTS/_audio_formats.py',
        'RealtimeTTS/alignment/evidence.py',
        'RealtimeTTS/alignment/windows.py',
    })
    missing = sorted(required_files - names)
    if missing:
        raise SystemExit(f"wheel is missing required runtime files: {', '.join(missing)}")
    if any(name == "tests" or name.startswith("tests/") or "/tests/" in name for name in names):
        raise SystemExit("wheel unexpectedly contains the test suite")
    dist_info_names = [name for name in names if ".dist-info/" in name]
    if not any(name.endswith("/METADATA") for name in dist_info_names):
        raise SystemExit("wheel does not contain dist-info metadata")
    license_names = {
        Path(name).name
        for name in dist_info_names
        if Path(name).name in {"LICENSE", "LICENSING_ADDENDUM.md"}
    }
    missing_licenses = {"LICENSE", "LICENSING_ADDENDUM.md"} - license_names
    if missing_licenses:
        raise SystemExit(
            "wheel is missing dist-info license files: "
            + ", ".join(sorted(missing_licenses))
        )
    with zipfile.ZipFile(wheel) as archive:
        if _CATALOG_PATH not in names:
            raise SystemExit(f"wheel is missing the bundled voice catalog: {_CATALOG_PATH}")
        expected = _catalogued_voice_hashes(archive.read(_CATALOG_PATH))
        _validate_catalogued_assets(names, expected, archive.read, "wheel")


def _check_sdist_contents(sdist: Path) -> None:
    with tarfile.open(sdist, mode="r:gz") as archive:
        names = {
            "/".join(Path(member.name).parts[1:]).replace("\\", "/")
            for member in archive.getmembers()
            if len(Path(member.name).parts) > 1
        }
    required_files = {
        "LICENSE",
        "LICENSING_ADDENDUM.md",
        "CHANGELOG.md",
        "README.md",
        "pyproject.toml",
        "setup.py",
        "RealtimeTTS/_version.py",
        "RealtimeTTS/qwen_server.py",
        "RealtimeTTS/qwen_emotions.py",
        "RealtimeTTS/engines/qwen_cpu_engine.py",
        "RealtimeTTS/engines/qwen_engine.py",
        "RealtimeTTS/engines/inflect_engine.py",
        "RealtimeTTS/engines/higgs_engine.py",
        "RealtimeTTS/alignment/__init__.py",
    }
    required_files.update({
        'RealtimeTTS/engines/breeze_engine.py',
        'RealtimeTTS/engines/speechify_engine.py',
        'RealtimeTTS/breeze_server.py',
        'RealtimeTTS/breeze_diagnostics.py',
        'RealtimeTTS/text_exchange.py',
        'RealtimeTTS/alignment/streaming.py',
        'RealtimeTTS/alignment/onnx_ctc.py',
        'RealtimeTTS/alignment/onnx_parakeet.py',
        'RealtimeTTS/alignment/timeline.py',
        'RealtimeTTS/alignment/controller.py',
        'RealtimeTTS/alignment/backends.py',
        'RealtimeTTS/alignment/decoder.py',
        'RealtimeTTS/alignment/server.py',
        'RealtimeTTS/server.py',
        'RealtimeTTS/breeze_worker.py',
        'RealtimeTTS/_audio_formats.py',
        'RealtimeTTS/alignment/evidence.py',
        'RealtimeTTS/alignment/windows.py',
        'docs/universal-alignment.md',
        'tools/prepare_breeze_cpp.py',
        'tools/breeze_cpp_controls.json',
        'docs/engines/breeze.md',
    })
    missing = sorted(required_files - names)
    if missing:
        raise SystemExit(f"sdist is missing required release files: {', '.join(missing)}")
    if any(name.startswith("tests/") or "/tests/" in name for name in names):
        raise SystemExit("sdist unexpectedly contains the test suite")
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = {
            "/".join(Path(member.name).parts[1:]).replace(chr(92), "/"): member
            for member in archive.getmembers()
            if len(Path(member.name).parts) > 1
        }
        if _CATALOG_PATH not in members:
            raise SystemExit(f"sdist is missing the bundled voice catalog: {_CATALOG_PATH}")
        catalog_member = archive.extractfile(members[_CATALOG_PATH])
        if catalog_member is None:
            raise SystemExit(f"sdist voice catalog is not a regular file: {_CATALOG_PATH}")
        expected = _catalogued_voice_hashes(catalog_member.read())

        def read_member(name: str) -> bytes:
            handle = archive.extractfile(members[name])
            if handle is None:
                raise SystemExit(f"sdist catalogued voice is not a regular file: {name}")
            return handle.read()

        _validate_catalogued_assets(names, expected, read_member, "sdist")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="smoke-test a RealtimeTTS wheel in a clean disposable venv"
    )
    parser.add_argument(
        "artifact",
        nargs="?",
        default="dist",
        help="wheel path or directory containing exactly one wheel (default: dist)",
    )
    args = parser.parse_args()

    sdist = _sdist_from_argument(args.artifact)
    wheel = _wheel_from_argument(args.artifact)
    _check_wheel_contents(wheel)
    if sdist is not None:
        _check_sdist_contents(sdist)

    with tempfile.TemporaryDirectory(prefix="realtimetts-clean-install-") as directory:
        environment = Path(directory) / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        interpreter = _venv_python(environment)

        _run(
            [
                str(interpreter),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--force-reinstall",
                str(wheel),
            ]
        )
        _run(
            [
                str(interpreter),
                "-c",
                "\n".join(
                    [
                        "import importlib.metadata as metadata",
                        "from pathlib import Path",
                        "import RealtimeTTS",
                        "dist = metadata.distribution('realtimetts')",
                        "assert dist.version == RealtimeTTS.__version__, (dist.version, RealtimeTTS.__version__, RealtimeTTS.__file__)",
                        "assert RealtimeTTS.__version__",
                        "assert Path(RealtimeTTS.__file__).is_file()",
                        "extras = set(dist.metadata.get_all('Provides-Extra') or [])",
                        "assert {'qwen-server', 'qwen-cpu-server', 'qwen-cpu', 'inflect', 'higgs', 'alignment', 'omniasr'} <= extras",
                        "assert {'breeze', 'breeze-server', 'speechify', 'alignment-cpu', 'alignment-multilingual'} <= extras",
                        "assert dist.metadata['Requires-Python'].replace(' ', '') == '>=3.10,<3.15'",
                        "assert any(ep.name == 'realtimetts-qwen-server' for ep in dist.entry_points)",
                        "assert any(ep.name == 'realtimetts-breeze-server' for ep in dist.entry_points)",
                        "import RealtimeTTS.qwen_emotions as demo",
                        "assert demo.main(['--list']) == 0",
                        "assert (Path(RealtimeTTS.__file__).parent / '_version.py').is_file()",
                    ]
                ),
            ],
            cwd=environment,
        )

    checked = wheel.name if sdist is None else f"{wheel.name}, {sdist.name}"
    print(f"clean-install smoke passed: {checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
