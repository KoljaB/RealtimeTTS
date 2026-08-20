"""Release metadata and import-boundary regression tests."""

from __future__ import annotations

import re
import subprocess
import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _version_from_source() -> str:
    text = (ROOT / "RealtimeTTS" / "_version.py").read_text(encoding="utf-8")
    match = re.fullmatch(
        r'(?s)""".*?"""\s*\n\s*__version__\s*=\s*"([^"]+)"\s*\n',
        text,
    )
    assert match, "_version.py must contain one parseable __version__ assignment"
    return match.group(1)


def test_release_version_is_single_source_and_current_candidate():
    assert _version_from_source() == "0.7.4.dev8"

    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "_version.py" in setup_text
    assert "version=current_version" in setup_text
    assert not re.search(r'current_version\s*=\s*["\']0\.7\.4["\']', setup_text)


def test_pep517_build_backend_and_pytest_collection_policy_are_declared():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires = ["setuptools>=68", "wheel"]' in text
    assert 'build-backend = "setuptools.build_meta"' in text
    assert 'testpaths = ["tests"]' in text
    assert 'python_files = ["test_*.py"]' in text


def test_setup_reports_source_version_from_outside_repository_root(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "setup.py"), "--version"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().splitlines()[-1] == _version_from_source()


def test_source_distribution_manifest_keeps_release_metadata_and_excludes_tests():
    lines = {
        line.strip()
        for line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert {"include LICENSE", "include LICENSING_ADDENDUM.md", "include CHANGELOG.md"} <= lines
    assert "include README.md" in lines
    assert "include requirements.txt" in lines
    assert "prune tests" in lines


def _run_isolated_import(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_root_voice_export_works_when_voice_is_imported_first():
    _run_isolated_import(
        """
import sys, types
sys.modules['pyaudio'] = types.ModuleType('pyaudio')
sys.modules['requests'] = types.ModuleType('requests')
from RealtimeTTS import ModelsLabVoice
assert ModelsLabVoice.__name__ == 'ModelsLabVoice'
"""
    )


def test_engines_voice_export_works_when_voice_is_imported_first():
    _run_isolated_import(
        """
import sys, types
sys.modules['pyaudio'] = types.ModuleType('pyaudio')
sys.modules['requests'] = types.ModuleType('requests')
from RealtimeTTS.engines import ModelsLabVoice
assert ModelsLabVoice.__name__ == 'ModelsLabVoice'
"""
    )


def test_root_exports_modelslab_symbols():
    text = (ROOT / "RealtimeTTS" / "__init__.py").read_text(encoding="utf-8")
    assert '"ModelsLabEngine", "ModelsLabVoice"' in text
