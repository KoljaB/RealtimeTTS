from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from RealtimeTTS.language_router import (
    FASTTEXT_TO_QWEN_LANGUAGE,
    FastTextLanguageDetector,
    LanguageDetection,
    QwenLanguageRouter,
    SUPPORTED_QWEN_LANGUAGES,
    language_lookahead_wait_ms,
    normalize_qwen_language,
)


class FakeFastTextModel:
    def __init__(self, labels=("__label__de",), probabilities=(0.98,)):
        self.labels = labels
        self.probabilities = probabilities
        self.calls = []

    def predict(self, text, k=1):
        self.calls.append((text, k))
        return self.labels, self.probabilities


class FakeClock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def test_fasttext_label_maps_to_qwen_language_and_reports_latency():
    model = FakeFastTextModel(("__label__de",), (0.97,))
    detector = FastTextLanguageDetector(
        model=model,
        clock=FakeClock(1_000_000_000, 1_000_250_000),
    )

    result = detector.detect("Guten Tag, wie geht es dir?")

    assert result.language == "german"
    assert result.label == "de"
    assert result.confidence == pytest.approx(0.97)
    assert result.latency_ms == pytest.approx(0.25)
    assert model.calls == [("Guten Tag, wie geht es dir?", 2)]


def test_explicit_language_bypasses_detector_and_routes_variant():
    class DetectorThatMustNotRun:
        def detect(self, _text):
            raise AssertionError("explicit language must bypass auto detection")

    router = QwenLanguageRouter(
        DetectorThatMustNotRun(),
        voice_routes={
            "mira": {
                "german": "mira_v5_spark_de",
                "english": "mira_v5_spark",
            }
        },
    )

    result = router.resolve(
        "Hello there.",
        base_voice="mira",
        language="English",
    )

    assert result.language == "english"
    assert result.voice == "mira_v5_spark"
    assert result.detector_used is False
    assert result.detection_latency_ms == 0.0


@dataclass
class FakeDetector:
    language: str
    calls: list[str]
    warmup_calls: list[str]

    def detect(self, text):
        self.calls.append(text)
        return LanguageDetection(self.language, self.language[:2], 1.0, 0.4)

    def warmup(self, text):
        self.warmup_calls.append(text)
        return self.detect(text)


def test_auto_detection_routes_per_base_voice_and_exposes_latency():
    detector = FakeDetector("german", [], [])
    router = QwenLanguageRouter(
        detector,
        voice_routes={
            "mira": {"de": "mira_v5_spark_de", "en": "mira_v5_spark"},
            "narrator": {"german": "narrator_de"},
        },
    )

    mira = router.resolve("Guten Tag.", base_voice="mira", language="auto")
    narrator = router.resolve("Guten Tag.", base_voice="narrator", language="auto")

    assert mira.language == "german"
    assert mira.voice == "mira_v5_spark_de"
    assert mira.detector_used is True
    assert mira.detection_latency_ms == pytest.approx(0.4)
    assert narrator.voice == "narrator_de"
    assert detector.calls == ["Guten Tag.", "Guten Tag."]


def test_startup_warmup_loads_model_once_and_caches_prediction():
    model = FakeFastTextModel(("__label__en",), (0.91,))
    loads = []

    def loader(path):
        loads.append(path)
        return model

    detector = FastTextLanguageDetector(
        "D:/models/lid.176.ftz",
        model_loader=loader,
    )
    router = QwenLanguageRouter(detector)

    first = router.warmup("Warm up now.")
    second = router.warmup("This text must not be predicted again.")
    resolved = router.resolve("Hello.", base_voice="mira")

    assert first.language == second.language == "english"
    assert loads == [str(Path("D:/models/lid.176.ftz"))]
    assert len(model.calls) == 2  # one warmup plus the later auto request
    assert detector.warmed is True
    assert resolved.detection_latency_ms >= 0.0


def test_unknown_fasttext_label_uses_configured_supported_fallback():
    model = FakeFastTextModel(("__label__ar",), (0.88,))
    detector = FastTextLanguageDetector(model=model, fallback_language="German")

    result = detector.detect("مرحبا")

    assert result.language == "german"
    assert result.label == "ar"
    assert result.used_fallback is True


def test_short_or_low_margin_text_uses_per_call_fallback_language():
    short = FastTextLanguageDetector(
        model=FakeFastTextModel(("__label__es", "__label__en"), (0.29, 0.20))
    )
    close = FastTextLanguageDetector(
        model=FakeFastTextModel(("__label__de", "__label__en"), (0.80, 0.70))
    )

    short_result = short.detect("Ja.", fallback_language="German")
    close_result = close.detect(
        "Das ist ein ausreichend langer Satz.", fallback_language="English"
    )

    assert short_result.language == "german"
    assert short_result.used_fallback is True
    assert close_result.language == "english"
    assert close_result.used_fallback is True


def test_clear_supported_language_just_below_old_threshold_is_accepted():
    detector = FastTextLanguageDetector(
        model=FakeFastTextModel(("__label__it", "__label__en"), (0.743, 0.05))
    )

    result = detector.detect("Con chi condivideresti un’esperienza del genere?")

    assert result.language == "italian"
    assert result.used_fallback is False


@pytest.mark.parametrize(
    ("score", "margin", "accepted"),
    [
        (0.67, 0.20, True),
        (0.60, 0.261, False),
        (0.60, 0.263, True),
        (0.55, 0.305, False),
        (0.55, 0.307, True),
        (0.50, 0.349, False),
        (0.50, 0.350, True),
        (0.49, 0.490, False),
    ],
)
def test_continuous_margin_requirement(score, margin, accepted):
    runner_up = score - margin
    detector = FastTextLanguageDetector(
        model=FakeFastTextModel(("__label__it", "__label__en"), (score, runner_up))
    )

    result = detector.detect("Questo è un testo abbastanza lungo.")

    assert result.used_fallback is (not accepted)
    assert result.language == ("italian" if accepted else "english")


def test_uncertain_supported_label_exposes_bounded_dynamic_lookahead():
    detector = FastTextLanguageDetector(
        model=FakeFastTextModel(
            ("__label__de", "__label__tr"),
            (0.4669775068759918, 0.281007319688797),
        )
    )

    result = detector.detect("Ja, nat\u00fcrlich,")

    assert result.used_fallback is True
    assert result.candidate_language == "german"
    assert result.runner_up_label == "tr"
    assert result.confidence_margin == pytest.approx(0.18597018718719482)
    assert language_lookahead_wait_ms(result) == pytest.approx(45.61192512512208)


def test_lookahead_wait_increases_continuously_as_margin_closes():
    close = FastTextLanguageDetector(
        model=FakeFastTextModel(("__label__de", "__label__en"), (0.41, 0.40))
    ).detect("Das ist ein kurzer Text.")
    wide = FastTextLanguageDetector(
        model=FakeFastTextModel(("__label__de", "__label__en"), (0.41, 0.21))
    ).detect("Das ist ein kurzer Text.")

    assert language_lookahead_wait_ms(close) == pytest.approx(116.0)
    assert language_lookahead_wait_ms(wide) == pytest.approx(40.0)


def test_lookahead_ignores_labels_below_the_agreed_candidate_floor():
    detector = FastTextLanguageDetector(
        model=FakeFastTextModel(("__label__de", "__label__en"), (0.39, 0.38))
    )

    result = detector.detect("Das ist ein kurzer Text.")

    assert language_lookahead_wait_ms(result) is None


def test_qwen_mapping_is_complete_and_auto_is_not_a_native_language():
    assert set(FASTTEXT_TO_QWEN_LANGUAGE.values()) == SUPPORTED_QWEN_LANGUAGES
    assert normalize_qwen_language("de-DE") == "german"
    with pytest.raises(ValueError, match="routing mode"):
        normalize_qwen_language("auto")
    assert normalize_qwen_language("auto", allow_auto=True) == "auto"
