"""CPU and CUDA wheels may coexist without sharing import-package files."""
import sys
from types import SimpleNamespace

import pytest

from RealtimeTTS.engines.qwen_cpu_engine import QwenCpuEngine
from RealtimeTTS.engines import qwen_cpu_engine as cpu_module
from RealtimeTTS import qwen_emotions as demo


def test_cpu_prefers_namespaced_package_when_gpu_is_installed(monkeypatch):
    cpu = SimpleNamespace(CPU_ONLY=True)
    gpu = SimpleNamespace(CPU_ONLY=False)
    monkeypatch.setitem(sys.modules, "qwentts_cpp_cpu", cpu)
    monkeypatch.setitem(sys.modules, "qwentts_cpp", gpu)
    engine = object.__new__(QwenCpuEngine)
    assert engine._load_native_module() is cpu
    assert demo._native_module("cpu") is cpu
    assert demo._native_module("gpu") is gpu


def test_cpu_does_not_hide_broken_package_dependency(monkeypatch):
    def broken(name):
        raise ModuleNotFoundError("missing numpy", name="numpy")
    monkeypatch.setattr(cpu_module.importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError, match="missing numpy"):
        object.__new__(QwenCpuEngine)._load_native_module()


def test_cpu_extra_selects_different_distribution_than_gpu():
    from pathlib import Path
    setup = (Path(__file__).resolve().parents[1] / "setup.py").read_text()
    assert '"realtimetts-qwen-native-cpu==0.3.0"' in setup
    assert 'realtimetts-qwen-native[cuda12]==0.2.0' in setup
    assert '"realtimetts-qwen-native==0.2.0+cpu2"' not in setup
