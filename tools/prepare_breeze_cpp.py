"""Prepare HoppouAI/Breeze-TTS-2.cpp a0e177f for native RealtimeTTS controls.

All anchors are validated before any write. --check only validates that the
control patch is installed. Existing unrelated content is preserved.
"""
import argparse
import json
from pathlib import Path


def plan(root, changes):
    pending = []
    for change in changes:
        path = (root / change["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Patch target escapes checkout")
        original = path.read_text(encoding="utf-8")
        modified = original
        for before, after in change["replacements"]:
            if after in modified:
                continue
            if modified.count(before) != 1:
                raise ValueError(f"Unexpected source version/anchor in {path}")
            modified = modified.replace(before, after, 1)
        if modified != original:
            pending.append((path, original, modified))
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.source.resolve(strict=True)
    changes = json.loads(Path(__file__).with_name("breeze_cpp_controls.json").read_text(encoding="utf-8"))
    pending = plan(root, changes)
    if args.check:
        if pending:
            raise SystemExit(f"Native controls are missing in {len(pending)} files")
        print("Native control patch verified")
        return
    for path, original, modified in pending:
        backup = path.with_name(path.name + ".before-realtimetts-controls")
        if not backup.exists():
            with backup.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(original)
        temporary = path.with_name(path.name + ".realtimetts-controls.tmp")
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(modified)
        temporary.replace(path)
        print(path.relative_to(root))
    print(f"Prepared {len(pending)} files; rebuild breeze-server before starting it.")


if __name__ == "__main__":
    main()
