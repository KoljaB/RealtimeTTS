"""Repaired macOS libraries remain hashed artifacts, not source-parity inputs."""
import importlib.util
from pathlib import Path
import zipfile

import pytest


def test_cpu_mac_delocate_payload_does_not_relax_python_source_parity(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "cpu_release_guard", Path(__file__).parents[1] / "tools" / "release_guard.py"
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    profile = guard.COMPONENT_PROFILES["RealtimeTTSQwenNativeCPUPublic"]
    package = tmp_path / "src" / "qwentts_cpp_cpu"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VERSION = '0.3.0rc1'\n")
    specs = [{"package_dir": "qwentts_cpp_cpu", "source_package_dir": "src/qwentts_cpp_cpu", "runtime_module": "qwentts_cpp_cpu"}]
    wheels = []
    for platform in profile["required_wheel_platforms"]:
        wheel = tmp_path / f"realtimetts_qwen_native_cpu-0.3.0rc1-py3-none-{platform}.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("qwentts_cpp_cpu/__init__.py", "VERSION = '0.3.0rc1'\n")
            for group in profile["required_native_library_groups"][platform]:
                archive.writestr(group[0], b"b47728b-native")
            if platform.startswith("macosx"):
                archive.writestr("qwentts_cpp_cpu/.dylibs/libggml-cpu.0.17.0.dylib", b"repaired-native")
            archive.writestr("realtimetts_qwen_native_cpu-0.3.0rc1.dist-info/METADATA", "Name: realtimetts-qwen-native-cpu\nVersion: 0.3.0rc1\n")
            archive.writestr("realtimetts_qwen_native_cpu-0.3.0rc1.dist-info/WHEEL", f"Wheel-Version: 1.0\nTag: py3-none-{platform}\n")
        wheels.append(wheel)
    metadata = guard._wheel_metadata(wheels[0])
    guard._validate_platform_wheels(profile, metadata, wheels[0], wheels[1:], tmp_path, specs)
    with zipfile.ZipFile(wheels[-1], "a") as archive:
        archive.writestr("qwentts_cpp_cpu/unexpected.py", "CHANGED = True\n")
    with pytest.raises(guard.GuardError, match="unexpected.py"):
        guard._validate_platform_wheels(profile, metadata, wheels[0], wheels[1:], tmp_path, specs)
