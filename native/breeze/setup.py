from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from setuptools import find_namespace_packages, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from build_package import PACKAGE_NAME, worker_filename


def _stage() -> tuple[Path, str]:
    platform = os.environ.get("BREEZE_WHEEL_PLATFORM", "").strip()
    name = worker_filename(platform)
    value = os.environ.get("BREEZE_NATIVE_STAGE_DIR", "").strip()
    if not value:
        raise RuntimeError("BREEZE_NATIVE_STAGE_DIR is required")
    root = Path(value).resolve()
    payload = root / PACKAGE_NAME / "bin"
    if not (payload / name).is_file() or not (payload / "build-info.json").is_file():
        raise RuntimeError("run build_package.py to create a valid native stage")
    return root, name


class NativeBuildPy(build_py):
    def run(self) -> None:
        root, name = _stage()
        super().run()
        source = root / PACKAGE_NAME / "bin"
        target = Path(self.build_lib) / PACKAGE_NAME / "bin"
        target.mkdir(parents=True, exist_ok=True)
        for filename in (name, "build-info.json"):
            shutil.copy2(source / filename, target / filename)


class NativeBdistWheel(bdist_wheel):
    def finalize_options(self) -> None:
        platform = os.environ.get("BREEZE_WHEEL_PLATFORM", "").strip()
        worker_filename(platform)
        self.plat_name = platform
        super().finalize_options()
        self.plat_name = platform
        self.plat_name_supplied = True
        self.root_is_pure = True
        self.python_tag = "py3"


setup(
    name="realtimetts-breeze-native",
    version="0.1.0",
    description="Prebuilt RealtimeTTS Breeze native worker launcher",
    long_description=(PROJECT_DIR / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=["LICENSE"],
    packages=find_namespace_packages("src"),
    package_dir={"": "src"},
    package_data={"realtimetts_breeze_native": ["bin/*", "licenses/*.txt"]},
    python_requires=">=3.9",
    entry_points={"console_scripts": [
        "realtimetts-breeze-native=realtimetts_breeze_native.launcher:main"
    ]},
    cmdclass={"build_py": NativeBuildPy, "bdist_wheel": NativeBdistWheel},
    zip_safe=False,
)
