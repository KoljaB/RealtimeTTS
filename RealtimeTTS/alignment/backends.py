"""Adapters from existing alignment backends to the common stream service."""
from __future__ import annotations

import re

import numpy as np

from .base import CharacterAlignment, WordAlignment
from .windows import window_anchor
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

    def prepare(self, text, level):
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
        return normalized, targets, projection

    def measure(self, text, targets, audio, sample_rate, *, offset_seconds=0.0, anchor_seconds=0.0):
        backend = self.backend
        if hasattr(backend, "align"):
            characters, _words, _complete = backend.align(
                audio, sample_rate, text, offset_seconds=offset_seconds, anchor_seconds=anchor_seconds
            )
            return characters
        emissions = backend.emissions(audio, sample_rate)
        step = backend.frame_seconds
        positions = offset_seconds + np.arange(len(emissions)) * step
        keep = positions >= anchor_seconds - 1e-6
        return display_ctc_characters(
            emissions[keep], targets, text, positions[keep], backend.blank_id, step
        )

    def event(self, normalized, characters, projection, level):
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
            "characters": [vars(item) for item in characters],
        }
        if projection is not None:
            result = []
            projection(result.append)(event)
            event = result[0] if result else {"words": [], "characters": []}
        if level != "character":
            event["characters"] = []
        return event

    def align(self, text, audio, sample_rate, *, level="word"):
        normalized, targets, projection = self.prepare(text, level)
        characters = self.measure(normalized, targets, audio, sample_rate)
        return self.event(normalized, characters, projection, level)

    def create_cursor(self, *, level="word"):
        return CPUAlignmentCursor(self, level=level)


class CPUAlignmentCursor:
    """Per-fragment overlap state; model sessions remain shared and serial.

    Committed text is retained, while inference revisits roughly the last 3.2
    seconds. A 12-second window bounds work even when recognition cannot advance.
    """
    window_seconds = 12.0

    def __init__(self, model, *, level):
        self.model, self.level = model, level
        self.characters = []
        self.normalized = ""
        self.text_offset = self.audio_start = 0
        self.anchor = 0.0
        self.sample_rate = None

    def align(self, text, audio, sample_rate, *, audio_start=0):
        normalized, targets, projection = self.model.prepare(text, self.level)
        if self.sample_rate not in (None, sample_rate):
            raise ValueError("Alignment sample rate changed within a fragment")
        self.sample_rate = sample_rate
        if normalized[:self.text_offset] != self.normalized[:self.text_offset]:
            raise ValueError("Transcript changed inside a committed alignment prefix")
        self.normalized = normalized
        available = audio_start + len(audio)
        if self.audio_start < audio_start:
            raise RuntimeError("Alignment audio was discarded before its committed anchor")
        while True:
            end = min(available, self.audio_start + round(self.window_seconds * sample_rate))
            window = audio[self.audio_start - audio_start:end - audio_start]
            characters = self.model.measure(
                normalized[self.text_offset:], targets[self.text_offset:], window, sample_rate,
                offset_seconds=self.audio_start / sample_rate, anchor_seconds=self.anchor,
            )
            self.characters = [c for c in self.characters if c.index < self.text_offset] + [
                CharacterAlignment(c.character, c.start_time, c.end_time,
                                   c.index + self.text_offset, c.score) for c in characters
            ]
            offset, start, anchor = window_anchor(
                normalized, self.characters, sample_rate, self.model.backend.frame_seconds,
                end / sample_rate,
            )
            if offset < self.text_offset:
                offset, start, anchor = self.text_offset, self.audio_start, self.anchor
            start = max(start, self.audio_start)
            if end < available and start <= self.audio_start:
                raise RuntimeError("Alignment could not advance within its 12-second audio window")
            self.text_offset, self.audio_start, self.anchor = offset, start, anchor
            if end >= available:
                return self.model.event(normalized, self.characters, projection, self.level)


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
