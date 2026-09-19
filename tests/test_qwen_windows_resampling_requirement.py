"""Both Qwen installs avoid the verified Windows first-resample regression."""
import ast
from pathlib import Path

from packaging.requirements import Requirement


def test_qwen_backends_share_windows_only_numba_constraint():
    source = (Path(__file__).resolve().parents[1] / "setup.py").read_text()
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: node.value for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    audio = assignments["qwen_audio_requirements"]
    requirements = [Requirement(item.value) for item in audio.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)]
    constraint = next(item for item in requirements if item.name == "numba")
    assert "0.66.0" in constraint.specifier
    assert "0.67.0" not in constraint.specifier
    assert constraint.marker.evaluate({"sys_platform": "win32"})
    for platform in ("linux", "darwin"):
        assert not constraint.marker.evaluate({"sys_platform": platform})
    for name in ("qwen_common_requirements", "qwen_cpu_common_requirements"):
        assert any(isinstance(item, ast.Name) and item.id == "qwen_audio_requirements"
                   for item in ast.walk(assignments[name]))
