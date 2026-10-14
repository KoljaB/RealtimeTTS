from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def executable_path() -> Path:
    name = "breeze-server.exe" if os.name == "nt" else "breeze-server"
    path = Path(__file__).resolve().parent / "bin" / name
    if not path.is_file():
        raise FileNotFoundError(f"packaged Breeze worker is missing: {path}")
    if os.name != "nt" and not os.access(path, os.X_OK):
        raise PermissionError(f"packaged Breeze worker is not executable: {path}")
    return path


def worker_command(args=()):
    return [os.fspath(executable_path()), *[os.fspath(arg) for arg in args]]


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: realtimetts-breeze-native <breeze-server arguments...>", file=sys.stderr)
        return 2
    return subprocess.call(worker_command(args))
