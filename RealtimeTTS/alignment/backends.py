"""Adapters from existing alignment backends to the common stream service."""
from __future__ import annotations

import re

import numpy as np

from .base import WordAlignment
from .streaming import display_ctc_characters


_LANGUAGE_NAMES = {
    "en": "english", "de": "german", "es": "spanish", "fr": "french",
    "it": "italian", "pt": "portuguese", "pt-br": "portuguese", "ru": "russian",
}


class CPUAlignmentModel:
    """Use the existing CPU acoustic backends without a TTS-engine dependency."""

    supports_partial = True

    def __init__(self, backend, language=None):
        self.backend = backend
        self.language = _LANGUAGE_NAMES.get(language, language)

    def align(self, text, audio, sample_rate, *, level="word"):
        backend = self.backend
        supported = getattr(backend, "language", None)
        languages = getattr(backend, "languages", ())
        if supported and self.language and self.language != supported:
            raise ValueError(f"Alignment model supports {supported}, not {self.language}")
        if languages and self.language not in languages:
            raise ValueError(f"Alignment model does not support {self.language}")
        levels = getattr(backend, "timestamp_levels", ("word", "character"))
        if level not in levels:
            raise ValueError(f"Alignment model does not provide {level} timestamps")
        projection = None
        if callable(getattr(backend, "prepare_text", None)):
            text, projection = backend.prepare_text(text)
        normalized, targets = backend.encode(text)
        if hasattr(backend, "align"):
            characters, words, _complete = backend.align(
                audio, sample_rate, normalized, offset_seconds=0.0, anchor_seconds=0.0
            )
        else:
            emissions = backend.emissions(audio, sample_rate)
            step = backend.frame_seconds
            positions = np.arange(len(emissions)) * step
            characters = display_ctc_characters(
                emissions, targets, normalized, positions, backend.blank_id, step
            )
            by_index = {c.index: c for c in characters}
            words = []
            for match in re.finditer(r"\S+", normalized):
                covered = [by_index.get(i) for i in range(match.start(), match.end())]
                if covered and all(item is not None for item in covered):
                    scores = [item.score for item in covered if item.score is not None]
                    words.append(WordAlignment(
                        match.group(), covered[0].start_time, covered[-1].end_time,
                        min(scores) if scores else None,
                    ))
        event = {
            "type": "alignment",
            "words": [vars(item) for item in words],
            "characters": [vars(item) for item in characters] if level == "character" else [],
        }
        if projection is not None:
            result = []
            projection(result.append)(event)
            event = result[0] if result else {"words": [], "characters": []}
        return event


def make_alignment_model(config, engine=None):
    model = config.aligner
    if model is not None:
        if isinstance(model, CPUAlignmentModel):
            return model
        if hasattr(model, "backend") and hasattr(model, "start"):
            language = _LANGUAGE_NAMES.get(config.language, config.language)
            backend = model.backend
            if getattr(backend, "language", None) not in (None, language):
                backend = getattr(model, "multilingual_backend", None)
                if backend is None:
                    raise ValueError(f"No alignment backend for {config.language}")
            return CPUAlignmentModel(backend, config.language)
        return model
    if config.model_dir is not None:
        from .onnx_ctc import OnnxCTCBackend
        language = _LANGUAGE_NAMES.get(config.language, config.language)
        return CPUAlignmentModel(
            OnnxCTCBackend(config.model_dir, language=language), language
        )
    inherited = getattr(engine, "streaming_aligner", None)
    if inherited is not None:
        from dataclasses import replace
        return make_alignment_model(replace(config, aligner=inherited), engine)
    # Keep the existing explicit optional backend as the default fallback.
    # No model packages are imported merely by importing RealtimeTTS.
    from .omniasr import OmniASRCTCAligner
    return OmniASRCTCAligner()


def align_snapshot(model, text, audio, sample_rate, level):
    if isinstance(model, CPUAlignmentModel):
        return model.align(text, audio, sample_rate, level=level)
    if level == "character":
        method = getattr(model, "align_characters", None)
        if not callable(method):
            raise ValueError("Selected aligner does not provide character timestamps")
        characters = list(method(text, audio, sample_rate))
        # Character backends may expose a normalized transcript; word grouping
        # uses their own indices instead of inventing per-character durations.
        by_index = {item.index: item for item in characters}
        words = []
        normalized = "".join(
            by_index[index].character if index in by_index else " "
            for index in range(max(by_index, default=-1) + 1)
        )
        for match in re.finditer(r"\S+", normalized):
            covered = [by_index.get(i) for i in range(match.start(), match.end())]
            if covered and all(item is not None for item in covered):
                scores = [item.score for item in covered if item.score is not None]
                words.append(WordAlignment(
                    match.group(), covered[0].start_time, covered[-1].end_time,
                    min(scores) if scores else None,
                ))
    else:
        words = list(model.align_words(text, audio, sample_rate))
        characters = []
    return {
        "words": [vars(item) for item in words],
        "characters": [vars(item) for item in characters],
    }
