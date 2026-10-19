"""Breeze native releases preserve pinned source and exact binary provenance."""
import hashlib
import json
import zipfile
from pathlib import Path
import pytest
from tools import release_guard as guard


def test_breeze_native_manifest_rejects_modified_binary(tmp_path):
    profile = guard._component_profile("RealtimeTTSBreezeNativeWindows", publishing=True)
    wheel = tmp_path / "realtimetts_breeze_native-0.1.0rc1-py3-none-win_amd64.whl"
    binary = b"native test executable"
    info = dict(source_revision=profile["native_revision"], ggml_revision=profile["ggml_revision"],
                control_patch_sha256=profile["control_patch_sha256"], platform="win_amd64",
                native_binary=dict(name="breeze-server.exe", sha256=hashlib.sha256(binary).hexdigest()))
    def write(payload):
        with zipfile.ZipFile(wheel, "w") as archive:
            prefix = "realtimetts_breeze_native-0.1.0rc1.dist-info/"
            archive.writestr(prefix + "METADATA", "Name: realtimetts-breeze-native\nVersion: 0.1.0rc1\n")
            archive.writestr(prefix + "WHEEL", "Tag: py3-none-win_amd64\n")
            archive.writestr(profile["native_build_info"], json.dumps(info))
            archive.writestr("realtimetts_breeze_native/bin/breeze-server.exe", payload)
    write(binary)
    metadata = guard._wheel_metadata(wheel)
    guard._validate_platform_wheels(profile, metadata, wheel, [], tmp_path, [])
    write(binary + b"changed")
    with pytest.raises(guard.GuardError, match="binary hash mismatch"):
        guard._validate_platform_wheels(profile, metadata, wheel, [], tmp_path, [])
