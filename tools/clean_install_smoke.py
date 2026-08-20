#!/usr/bin/env python3
"""Install one built wheel into a disposable venv and verify its boundaries.

The smoke test intentionally installs with ``--no-deps`` and ``--no-index``.
It validates the artifact itself without downloading a model, an optional
engine, or any dependency from a package index.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path
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


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _check_wheel_contents(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    required_files = {
        "RealtimeTTS/_version.py",
        "RealtimeTTS/qwen_server.py",
        "RealtimeTTS/engines/qwen_engine.py",
        "RealtimeTTS/engines/inflect_engine.py",
    }
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
    forbidden_suffixes = (
        ".wav", ".flac", ".mp3", ".ogg", ".m4a",
        ".spk", ".rvq", ".gguf", ".safetensors", ".ckpt",
        ".pt", ".pth", ".onnx",
    )
    bundled_assets = sorted(
        name for name in names if name.lower().endswith(forbidden_suffixes)
    )
    if bundled_assets:
        raise SystemExit(
            "wheel unexpectedly bundles model or voice assets: "
            + ", ".join(bundled_assets)
        )


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
        "RealtimeTTS/engines/qwen_engine.py",
        "RealtimeTTS/engines/inflect_engine.py",
    }
    missing = sorted(required_files - names)
    if missing:
        raise SystemExit(f"sdist is missing required release files: {', '.join(missing)}")
    if any(name == "tests" or name.startswith("tests/") or "/tests/" in name for name in names):
        raise SystemExit("sdist unexpectedly contains the test suite")
    forbidden_suffixes = (
        ".wav", ".flac", ".mp3", ".ogg", ".m4a",
        ".spk", ".rvq", ".gguf", ".safetensors", ".ckpt",
        ".pt", ".pth", ".onnx",
    )
    bundled_assets = sorted(
        name for name in names if name.lower().endswith(forbidden_suffixes)
    )
    if bundled_assets:
        raise SystemExit(
            "sdist unexpectedly bundles model or voice assets: "
            + ", ".join(bundled_assets)
        )


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
                        "assert dist.version == RealtimeTTS.__version__",
                        "assert RealtimeTTS.__version__",
                        "assert Path(RealtimeTTS.__file__).is_file()",
                        "extras = set(dist.metadata.get_all('Provides-Extra') or [])",
                        "assert {'qwen-server', 'inflect'} <= extras",
                        "assert any(ep.name == 'realtimetts-qwen-server' for ep in dist.entry_points)",
                        "assert (Path(RealtimeTTS.__file__).parent / '_version.py').is_file()",
                    ]
                ),
            ]
        )

    checked = wheel.name if sdist is None else f"{wheel.name}, {sdist.name}"
    print(f"clean-install smoke passed: {checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
