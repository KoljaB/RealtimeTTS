from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native" / "breeze"))
sys.path.insert(0, str(ROOT / "native" / "breeze" / "src"))

import build_package
from realtimetts_breeze_native import launcher


def _cache_text(source: Path, overrides: Optional[dict[str, str]] = None) -> str:
    values = {
        "CMAKE_HOME_DIRECTORY": source.as_posix(),
        "CMAKE_BUILD_TYPE": "Release",
        **build_package.REQUIRED_CACHE_FLAGS,
    }
    values.update(overrides or {})
    return "\n".join(
        f"{key}:STRING={value}" for key, value in sorted(values.items())
    ) + "\n"


@pytest.fixture()
def native_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    source = tmp_path / "source"
    (source / "third_party" / "ggml").mkdir(parents=True)
    build = tmp_path / "build"
    build.mkdir()
    patch = tmp_path / "controls.json"
    patch.write_text("test controls\n", encoding="utf-8")
    (build / "CMakeCache.txt").write_text(_cache_text(source), encoding="utf-8")
    binary = build / "breeze-server.exe"
    binary.write_bytes(b"native-test-binary")
    revisions = {
        source.resolve(): build_package.UPSTREAM_REVISION,
        (source / "third_party" / "ggml").resolve(): build_package.GGML_REVISION,
    }

    def fake_git(path: Path, expected: str) -> None:
        if revisions[path.resolve()] != expected:
            raise build_package.BuildInputError("revision mismatch")

    monkeypatch.setattr(build_package, "CONTROL_PATCH", patch)
    monkeypatch.setattr(
        build_package,
        "CONTROL_PATCH_SHA256",
        hashlib.sha256(patch.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(build_package, "_git", fake_git)
    monkeypatch.setattr(build_package, "_check_patch", lambda source: None)
    return {"source": source, "build": build, "binary": binary}


def test_missing_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "BREEZE_SOURCE_DIR",
        "BREEZE_BUILD_DIR",
        "BREEZE_NATIVE_STAGE_DIR",
        "BREEZE_WHEEL_PLATFORM",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(build_package.BuildInputError, match="BREEZE_SOURCE_DIR"):
        build_package.main()


def test_wrong_revision_is_rejected(
    native_fixture: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        build_package,
        "_git",
        lambda path, expected: (_ for _ in ()).throw(
            build_package.BuildInputError("source revision mismatch")
        ),
    )
    with pytest.raises(build_package.BuildInputError, match="source revision"):
        build_package.assemble(
            native_fixture["source"],
            native_fixture["build"],
            native_fixture["build"] / "stage",
            "win_amd64",
        )


def test_cmake_source_and_flags_are_checked(
    native_fixture: dict[str, Path],
) -> None:
    cache = native_fixture["build"] / "CMakeCache.txt"
    cache.write_text(
        _cache_text(native_fixture["source"] / "other"),
        encoding="utf-8",
    )
    with pytest.raises(build_package.BuildInputError, match="CMAKE_HOME_DIRECTORY"):
        build_package.assemble(
            native_fixture["source"],
            native_fixture["build"],
            native_fixture["build"] / "stage",
            "win_amd64",
        )
    cache.write_text(
        _cache_text(native_fixture["source"], {"BREEZE_MP3": "ON"}),
        encoding="utf-8",
    )
    with pytest.raises(build_package.BuildInputError, match="BREEZE_MP3"):
        build_package.assemble(
            native_fixture["source"],
            native_fixture["build"],
            native_fixture["build"] / "stage",
            "win_amd64",
        )


def test_stage_contains_provenance_and_hash(
    native_fixture: dict[str, Path], tmp_path: Path
) -> None:
    destination = build_package.assemble(
        native_fixture["source"],
        native_fixture["build"],
        tmp_path / "stage",
        "win_amd64",
    )
    info = json.loads(destination.with_name("build-info.json").read_text(encoding="utf-8"))
    assert info["source_revision"] == build_package.UPSTREAM_REVISION
    assert info["ggml_revision"] == build_package.GGML_REVISION
    assert info["native_binary"]["sha256"] == hashlib.sha256(
        native_fixture["binary"].read_bytes()
    ).hexdigest()
    assert info["cmake"]["source_matches"]


def test_manylinux_is_not_an_assembler_platform() -> None:
    with pytest.raises(build_package.BuildInputError, match="native wheel platform"):
        build_package.worker_filename("manylinux_2_35_x86_64")


def test_launcher_keeps_invocation_shell_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(launcher, "executable_path", lambda: Path("breeze-server.exe"))
    monkeypatch.setattr(
        launcher.subprocess,
        "call",
        lambda command: calls.append(command) or 17,
    )
    assert launcher.main(["--model", "voice.gguf"]) == 17
    assert calls == [["breeze-server.exe", "--model", "voice.gguf"]]
