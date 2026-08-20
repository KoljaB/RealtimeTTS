"""Local language identification and Qwen voice routing.

The Qwen HTTP server can use :class:`QwenLanguageRouter` at the point where a
complete sentence is handed to TTS.  ``fasttext`` is deliberately imported
only when a detector without an injected model is used, so importing this
module does not make the optional language-identification dependency
mandatory.

The router has no dependency on a Qwen engine or on the HTTP server.  That is
intentional: callers can resolve a language and a voice before taking the
engine lock, while tests can inject a tiny deterministic detector/model.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Union


AUTO_LANGUAGE = "auto"
DEFAULT_WARMUP_TEXT = "This is a local language detection warmup sentence."
LANGUAGE_LOOKAHEAD_MIN_CONFIDENCE = 0.40
LANGUAGE_LOOKAHEAD_FULL_MARGIN = 0.20
LANGUAGE_LOOKAHEAD_MIN_MS = 40.0
LANGUAGE_LOOKAHEAD_MAX_MS = 120.0

# Qwen3-TTS supports these ten languages.  Values intentionally use the
# lowercase spelling used by ``QwenVoice`` and by qwentts.cpp's ``lang``
# request field in this repository.
SUPPORTED_QWEN_LANGUAGES = frozenset(
    {
        "chinese",
        "english",
        "japanese",
        "korean",
        "german",
        "french",
        "russian",
        "portuguese",
        "spanish",
        "italian",
    }
)

# fastText lid.176 labels are ISO-639-1 codes.  Keep this mapping explicit so
# unsupported lid labels cannot accidentally be sent to Qwen as if they were
# valid Qwen language names.
FASTTEXT_TO_QWEN_LANGUAGE: Mapping[str, str] = MappingProxyType(
    {
        "zh": "chinese",
        "en": "english",
        "ja": "japanese",
        "ko": "korean",
        "de": "german",
        "fr": "french",
        "ru": "russian",
        "pt": "portuguese",
        "es": "spanish",
        "it": "italian",
    }
)


_QWEN_LANGUAGE_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        # Chinese
        "zh": "chinese",
        "zh-cn": "chinese",
        "zh-hans": "chinese",
        "cmn": "chinese",
        "chinese": "chinese",
        # English
        "en": "english",
        "en-gb": "english",
        "en-us": "english",
        "english": "english",
        # Japanese
        "ja": "japanese",
        "jp": "japanese",
        "japanese": "japanese",
        # Korean
        "ko": "korean",
        "kr": "korean",
        "korean": "korean",
        # German
        "de": "german",
        "de-de": "german",
        "deutsch": "german",
        "german": "german",
        # French
        "fr": "french",
        "fr-fr": "french",
        "french": "french",
        # Russian
        "ru": "russian",
        "ru-ru": "russian",
        "russian": "russian",
        # Portuguese
        "pt": "portuguese",
        "pt-br": "portuguese",
        "pt-pt": "portuguese",
        "portuguese": "portuguese",
        # Spanish
        "es": "spanish",
        "es-es": "spanish",
        "spanish": "spanish",
        # Italian
        "it": "italian",
        "it-it": "italian",
        "italian": "italian",
    }
)


def normalize_qwen_language(value: str, *, allow_auto: bool = False) -> str:
    """Return the canonical lowercase Qwen language name for ``value``.

    ``value`` may be a Qwen language name, a fastText/ISO code, or a common
    locale alias such as ``de-DE``.  ``auto`` is accepted only when the caller
    explicitly opts in with ``allow_auto=True``; this prevents accidentally
    passing the sentinel into native synthesis.
    """

    if not isinstance(value, str):
        raise TypeError("language must be a string")
    normalized = value.strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("language must not be empty")
    if normalized == AUTO_LANGUAGE:
        if allow_auto:
            return AUTO_LANGUAGE
        raise ValueError("'auto' is a routing mode, not a Qwen language")
    language = _QWEN_LANGUAGE_ALIASES.get(normalized)
    if language is None or language not in SUPPORTED_QWEN_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_QWEN_LANGUAGES))
        raise ValueError(f"unsupported Qwen language {value!r}; supported: {supported}")
    return language


class FastTextModel(Protocol):
    """Minimal fastText model interface used by the detector."""

    def predict(self, text: str, k: int = 1) -> tuple[Sequence[Any], Sequence[Any]]:
        ...


@dataclass(frozen=True)
class LanguageDetection:
    """One local language-identification result.

    ``language`` is always a canonical Qwen language.  ``label`` is the raw
    normalized fastText label (for example ``en``), or ``None`` when a caller
    supplied a detector that returns only a Qwen language.  ``latency_ms`` is
    measured around the detector call with ``perf_counter_ns`` and is useful
    for server metrics without adding another clock in the request path.
    """

    language: str
    label: Optional[str]
    confidence: Optional[float]
    latency_ms: float
    used_fallback: bool = False
    candidate_language: Optional[str] = None
    runner_up_label: Optional[str] = None
    runner_up_confidence: Optional[float] = None
    confidence_margin: Optional[float] = None


def language_lookahead_wait_ms(detection: LanguageDetection) -> Optional[float]:
    """Return the bounded first-fragment wait for an uncertain supported label."""

    if (
        not detection.used_fallback
        or detection.candidate_language is None
        or detection.confidence is None
        or detection.confidence < LANGUAGE_LOOKAHEAD_MIN_CONFIDENCE
    ):
        return None
    margin = max(0.0, float(detection.confidence_margin or 0.0))
    uncertainty = 1.0 - min(1.0, margin / LANGUAGE_LOOKAHEAD_FULL_MARGIN)
    return LANGUAGE_LOOKAHEAD_MIN_MS + (
        LANGUAGE_LOOKAHEAD_MAX_MS - LANGUAGE_LOOKAHEAD_MIN_MS
    ) * uncertainty


class FastTextLanguageDetector:
    """Lazy, reusable wrapper around a local ``lid.176.ftz`` model.

    The normal production construction is ``FastTextLanguageDetector(path)``.
    ``model`` and ``model_loader`` are injectable so unit tests and embedders
    can avoid importing fastText or downloading a model.  The model is loaded
    once and prediction is serialized because the Python fastText binding does
    not promise concurrent access to a model object.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        *,
        model: Optional[FastTextModel] = None,
        model_loader: Optional[Callable[[str], FastTextModel]] = None,
        fallback_language: str = "english",
        minimum_characters: int = 12,
        minimum_words: int = 3,
        minimum_confidence: float = 0.50,
        full_confidence: float = 0.67,
        minimum_margin: float = 0.20,
        maximum_margin: float = 0.35,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if model is None and model_path is None:
            raise ValueError("model_path or an injected model is required")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.model_path = Path(model_path).expanduser() if model_path is not None else None
        self._model = model
        self._model_loader = model_loader
        self.fallback_language = normalize_qwen_language(fallback_language)
        self.minimum_characters = int(minimum_characters)
        self.minimum_words = int(minimum_words)
        self.minimum_confidence = float(minimum_confidence)
        self.full_confidence = float(full_confidence)
        self.minimum_margin = float(minimum_margin)
        self.maximum_margin = float(maximum_margin)
        if not 0.0 <= self.minimum_confidence < self.full_confidence <= 1.0:
            raise ValueError(
                "confidence thresholds must satisfy "
                "0 <= minimum_confidence < full_confidence <= 1"
            )
        if not 0.0 <= self.minimum_margin <= self.maximum_margin <= 1.0:
            raise ValueError(
                "margin thresholds must satisfy "
                "0 <= minimum_margin <= maximum_margin <= 1"
            )
        self._clock = clock
        self._model_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._warmup_lock = threading.Lock()
        self._warmed = False
        self._warmup_detection: Optional[LanguageDetection] = None
        self.last_detection: Optional[LanguageDetection] = None

    @property
    def model_loaded(self) -> bool:
        """Whether the model has been loaded (or was injected) already."""

        return self._model is not None

    @property
    def warmed(self) -> bool:
        return self._warmed

    def _load_model(self) -> FastTextModel:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            if self._model_loader is None:
                try:
                    import fasttext  # type: ignore[import-not-found]
                except ImportError as exc:  # pragma: no cover - exercised by integration users
                    raise ImportError(
                        "FastTextLanguageDetector requires the optional 'fasttext' package; "
                        "install it and provide a local lid.176.ftz model"
                    ) from exc
                self._model_loader = fasttext.load_model
            if self.model_path is None:
                raise ValueError("model_path is required when no model is injected")
            self._model = self._model_loader(str(self.model_path))
            return self._model

    @staticmethod
    def _label_text(label: Any) -> str:
        if isinstance(label, bytes):
            label = label.decode("utf-8", errors="replace")
        return str(label).strip().lower()

    @classmethod
    def _fasttext_code(cls, label: Any) -> str:
        normalized = cls._label_text(label)
        if normalized.startswith("__label__"):
            normalized = normalized[len("__label__") :]
        return normalized.replace("_", "-")

    @staticmethod
    def _first(value: Any) -> Optional[Any]:
        if isinstance(value, (str, bytes)):
            return value
        try:
            return next(iter(value))
        except (TypeError, StopIteration):
            return None

    def detect(
        self, text: str, *, fallback_language: Optional[str] = None
    ) -> LanguageDetection:
        """Classify non-empty text and map its label to a Qwen language."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        model = self._load_model()
        with self._predict_lock:
            started_ns = self._clock()
            labels, probabilities = model.predict(text.strip(), k=2)
            elapsed_ns = self._clock() - started_ns

        label_values = list(labels) if not isinstance(labels, (str, bytes)) else [labels]
        probability_values = (
            list(probabilities)
            if not isinstance(probabilities, (str, bytes))
            else [probabilities]
        )
        raw_label = self._first(label_values)
        label = self._fasttext_code(raw_label) if raw_label is not None else None
        detected = FASTTEXT_TO_QWEN_LANGUAGE.get(label or "")
        raw_runner_up_label = label_values[1] if len(label_values) > 1 else None
        runner_up_label = (
            self._fasttext_code(raw_runner_up_label)
            if raw_runner_up_label is not None
            else None
        )
        raw_probability = self._first(probability_values)
        try:
            confidence = float(raw_probability) if raw_probability is not None else None
        except (TypeError, ValueError):
            confidence = None
        try:
            runner_up = float(probability_values[1]) if len(probability_values) > 1 else 0.0
        except (TypeError, ValueError):
            runner_up = 0.0
        normalized_text = text.strip()
        words = normalized_text.split()
        margin = confidence - runner_up if confidence is not None else -1.0
        if confidence is None:
            required_margin = self.maximum_margin
        else:
            interpolation = min(
                1.0,
                max(
                    0.0,
                    (self.full_confidence - confidence)
                    / (self.full_confidence - self.minimum_confidence),
                ),
            )
            required_margin = self.minimum_margin + interpolation * (
                self.maximum_margin - self.minimum_margin
            )
        strong_enough = (
            detected is not None
            and confidence is not None
            and (len(normalized_text) >= self.minimum_characters or len(words) >= self.minimum_words)
            and confidence >= self.minimum_confidence
            and margin >= required_margin
        )
        used_fallback = not strong_enough
        fallback = (
            normalize_qwen_language(fallback_language)
            if fallback_language is not None
            else self.fallback_language
        )
        result = LanguageDetection(
            language=detected if strong_enough else fallback,
            label=label,
            confidence=confidence,
            latency_ms=max(0.0, float(elapsed_ns) / 1_000_000.0),
            used_fallback=used_fallback,
            candidate_language=detected,
            runner_up_label=runner_up_label,
            runner_up_confidence=runner_up,
            confidence_margin=margin if confidence is not None else None,
        )
        self.last_detection = result
        return result

    def warmup(self, text: str = DEFAULT_WARMUP_TEXT) -> LanguageDetection:
        """Load the model and run one prediction before serving requests.

        The prediction is cached, so repeated startup hooks do not perform
        extra work.  The returned result includes the warmup classification and
        its measurable latency.
        """

        with self._warmup_lock:
            if self._warmed and self._warmup_detection is not None:
                return self._warmup_detection
            result = self.detect(text)
            self._warmup_detection = result
            self._warmed = True
            return result


@dataclass(frozen=True)
class LanguageRoute:
    """Resolved Qwen language and voice for one synthesis sentence."""

    base_voice: str
    voice: str
    language: str
    detection: Optional[LanguageDetection]

    @property
    def detector_used(self) -> bool:
        return self.detection is not None

    @property
    def detection_latency_ms(self) -> float:
        return self.detection.latency_ms if self.detection is not None else 0.0


class QwenLanguageRouter:
    """Resolve ``auto``/explicit language and per-base-voice variants.

    ``voice_routes`` is shaped like ``{"mira": {"german":
    "mira_v5_spark_de", "english": "mira_v5_spark"}}``.  A language without
    a variant falls back to the base voice, allowing a single-language voice
    to remain usable while multilingual variants are added incrementally.
    """

    def __init__(
        self,
        detector: Any,
        *,
        voice_routes: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> None:
        if detector is None or not callable(getattr(detector, "detect", None)):
            raise TypeError("detector must provide detect(text)")
        self.detector = detector
        self._voice_routes = self._normalize_voice_routes(voice_routes or {})

    @staticmethod
    def _normalize_voice_routes(
        routes: Mapping[str, Mapping[str, str]],
    ) -> dict[str, dict[str, str]]:
        normalized: dict[str, dict[str, str]] = {}
        for raw_base_voice, raw_languages in routes.items():
            base_voice = str(raw_base_voice).strip()
            if not base_voice:
                raise ValueError("voice route base names must not be empty")
            if not isinstance(raw_languages, Mapping):
                raise TypeError(f"voice routes for {base_voice!r} must be a mapping")
            language_routes: dict[str, str] = {}
            for raw_language, raw_voice in raw_languages.items():
                route_voice = str(raw_voice).strip()
                if not route_voice:
                    raise ValueError(f"voice route for {base_voice!r} must not be empty")
                language_key = str(raw_language).strip().lower()
                if language_key in {"*", "default"}:
                    language_key = "default"
                else:
                    language_key = normalize_qwen_language(language_key)
                language_routes[language_key] = route_voice
            normalized[base_voice] = language_routes
        return normalized

    @property
    def voice_routes(self) -> Mapping[str, Mapping[str, str]]:
        """Read-only-by-convention copy of configured voice routes."""

        return {base: dict(routes) for base, routes in self._voice_routes.items()}

    def voice_for(self, base_voice: str, language: str) -> str:
        """Return the configured variant or ``base_voice`` as a fallback."""

        base = str(base_voice).strip()
        if not base:
            raise ValueError("base_voice must not be empty")
        canonical_language = normalize_qwen_language(language)
        routes = self._voice_routes.get(base)
        if not routes:
            return base
        return routes.get(canonical_language, routes.get("default", base))

    def warmup(self, text: str = DEFAULT_WARMUP_TEXT) -> Any:
        """Delegate detector warmup to the server startup hook."""

        warmup = getattr(self.detector, "warmup", None)
        if callable(warmup):
            return warmup(text)
        # A small injected fake need only implement detect().  Production
        # FastTextLanguageDetector always provides the dedicated warmup method.
        return self.detector.detect(text)

    @staticmethod
    def _detector_result(value: Any) -> LanguageDetection:
        if isinstance(value, LanguageDetection):
            return value
        if isinstance(value, str):
            return LanguageDetection(
                language=normalize_qwen_language(value),
                label=None,
                confidence=None,
                latency_ms=0.0,
            )
        language = getattr(value, "language", None)
        if isinstance(language, str):
            return LanguageDetection(
                language=normalize_qwen_language(language),
                label=getattr(value, "label", None),
                confidence=getattr(value, "confidence", None),
                latency_ms=float(getattr(value, "latency_ms", 0.0)),
                used_fallback=bool(getattr(value, "used_fallback", False)),
            )
        raise TypeError("detector.detect(text) must return LanguageDetection or a language string")

    def resolve(
        self,
        text: str,
        *,
        base_voice: str,
        language: Optional[str] = AUTO_LANGUAGE,
        fallback_language: Optional[str] = None,
    ) -> LanguageRoute:
        """Resolve one sentence, bypassing detection for explicit languages."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        base = str(base_voice).strip()
        if not base:
            raise ValueError("base_voice must not be empty")

        requested = AUTO_LANGUAGE if language is None else str(language).strip().lower()
        if requested == AUTO_LANGUAGE:
            if fallback_language is not None and isinstance(
                self.detector, FastTextLanguageDetector
            ):
                detection = self._detector_result(
                    self.detector.detect(
                        text.strip(), fallback_language=fallback_language
                    )
                )
            else:
                detection = self._detector_result(self.detector.detect(text.strip()))
            selected_language = normalize_qwen_language(detection.language)
        else:
            # This branch intentionally never touches detector.detect().
            detection = None
            selected_language = normalize_qwen_language(requested)
        selected_voice = self.voice_for(base, selected_language)
        return LanguageRoute(
            base_voice=base,
            voice=selected_voice,
            language=selected_language,
            detection=detection,
        )


__all__ = [
    "AUTO_LANGUAGE",
    "DEFAULT_WARMUP_TEXT",
    "FASTTEXT_TO_QWEN_LANGUAGE",
    "FastTextLanguageDetector",
    "FastTextModel",
    "LanguageDetection",
    "LanguageRoute",
    "QwenLanguageRouter",
    "SUPPORTED_QWEN_LANGUAGES",
    "language_lookahead_wait_ms",
    "normalize_qwen_language",
]
