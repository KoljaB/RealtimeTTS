from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


class _FakeCuda:
    emptied = False

    @classmethod
    def is_available(cls):
        return False

    @classmethod
    def empty_cache(cls):
        cls.emptied = True


class _FakePipeline:
    def __init__(self, *, repo_id, lang_code):
        self.repo_id = repo_id
        self.lang_code = lang_code
        self.loaded = []

    def load_single_voice(self, name):
        self.loaded.append(name)
        value = float(len(self.loaded))
        return np.array([value, value + 1.0], dtype=np.float32)

    def __call__(self, text, *, voice, speed):
        return iter(())


def _load_kokoro_module(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.FloatTensor = object
    fake_torch.cuda = _FakeCuda
    fake_kokoro = types.ModuleType("kokoro")
    fake_kokoro.KPipeline = _FakePipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)
    monkeypatch.delitem(
        sys.modules,
        "RealtimeTTS.engines.kokoro_engine",
        raising=False,
    )
    return importlib.import_module("RealtimeTTS.engines.kokoro_engine")


def test_kokoro_blend_cache_is_bounded_and_rejects_unsafe_formulas(monkeypatch):
    module = _load_kokoro_module(monkeypatch)
    engine = module.KokoroEngine(max_blended_voice_cache=2)
    pipeline = engine._get_pipeline("a")

    first = engine._parse_mixed_voice_formula(
        "0.25*af_one + 0.75*af_two",
        pipeline,
    )
    cached = engine._parse_mixed_voice_formula(
        "0.25*af_one + 0.75*af_two",
        pipeline,
    )
    np.testing.assert_array_equal(cached, first)
    assert pipeline.loaded == ["af_one", "af_two"]

    engine._parse_mixed_voice_formula("1*af_three", pipeline)
    engine._parse_mixed_voice_formula("1*af_four", pipeline)
    assert list(engine.blended_voices) == ["1*af_three", "1*af_four"]

    with pytest.raises(ValueError, match="Mixed-language"):
        engine._parse_mixed_voice_formula("0.5*af_one + 0.5*jf_one", pipeline)
    with pytest.raises(ValueError, match="finite and non-negative"):
        engine._parse_mixed_voice_formula("-1*af_one", pipeline)
    with pytest.raises(ValueError, match="finite and non-negative"):
        engine._parse_mixed_voice_formula("nan*af_one", pipeline)


def test_kokoro_honors_explicit_language_and_fails_on_empty_output(monkeypatch):
    module = _load_kokoro_module(monkeypatch)
    voice = module.KokoroVoice("custom", language_code="j")
    engine = module.KokoroEngine(voice=voice)

    assert engine.current_voice == "custom"
    assert engine.current_lang == "j"
    assert engine.synthesize("No audio from the fake pipeline") is False
    assert engine.queue.empty()


@pytest.mark.parametrize(
    ("formula", "language"),
    [
        ("0.3 * hf_alpha + 0.7 * hm_omega", "h"),
        ("0.5* jf_alpha + 0.5* jf_nezumi", "j"),
    ],
)
def test_kokoro_detects_language_of_formula_with_spaces(monkeypatch, formula, language):
    module = _load_kokoro_module(monkeypatch)
    engine = module.KokoroEngine()

    engine.set_voice(formula)

    assert engine.current_lang == language
    pipeline = engine._get_pipeline(engine.current_lang)
    engine._parse_mixed_voice_formula(formula, pipeline)
    assert pipeline.lang_code == language
