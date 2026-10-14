"""Transcript coverage and an explicit acoustic policy for decoder stopping."""
from __future__ import annotations

from dataclasses import dataclass
import math
import unicodedata

import numpy as np


def _symbols(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def coverage_counts(text, values):
    """Count sequentially matched spoken symbols, ignoring punctuation/spacing."""
    expected = _symbols(text)
    cursor = covered = 0
    for value in values:
        needle = _symbols(value)
        position = expected.find(needle, cursor) if needle else -1
        if position >= 0:
            covered += len(needle)
            cursor = position + len(needle)
    return covered, len(expected)


@dataclass(frozen=True)
class CompletionPolicy:
    """Conservative defaults, not calibrated probabilities of correct speech.

    A full alignment is only a candidate. Stopping also requires acoustic
    support and a stable endpoint (or the existing tail allowance as context).
    """
    minimum_token_probability: float = 0.35
    minimum_mean_log_margin: float = 0.0
    minimum_supported_fraction: float = 0.8
    minimum_word_supported_fraction: float = 0.5
    word_support_minimum_length: int = 4
    minimum_boundary_probability: float = 0.9
    minimum_suffix_supported_fraction: float = 0.75
    stable_updates: int = 2
    endpoint_tolerance_seconds: float = 0.06

    def __post_init__(self):
        if not math.isfinite(self.minimum_token_probability) or not 0 < self.minimum_token_probability <= 1:
            raise ValueError("minimum_token_probability must be in (0, 1]")
        for name in ("minimum_supported_fraction", "minimum_word_supported_fraction",
                     "minimum_suffix_supported_fraction", "minimum_boundary_probability"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if (isinstance(self.word_support_minimum_length, bool)
                or not isinstance(self.word_support_minimum_length, int)
                or self.word_support_minimum_length < 1):
            raise ValueError("word_support_minimum_length must be a positive integer")
        if not math.isfinite(self.minimum_mean_log_margin):
            raise ValueError("minimum_mean_log_margin must be finite")
        if isinstance(self.stable_updates, bool) or not isinstance(self.stable_updates, int) or self.stable_updates < 1:
            raise ValueError("stable_updates must be a positive integer")
        if not math.isfinite(self.endpoint_tolerance_seconds) or self.endpoint_tolerance_seconds < 0:
            raise ValueError("endpoint_tolerance_seconds must be finite and nonnegative")

    def supports(self, characters, mean_log_margin=None, final_log_margin=None):
        words, word = [], []
        for character in characters:
            if character.character.isspace():
                if word:
                    words.append(word)
                    word = []
            elif any(c.isalnum() for c in character.character):
                word.append(character)
        if word:
            words.append(word)
        if not words:
            return False
        supported = lambda c: (c.score is not None and math.isfinite(c.score)
                               and c.score >= self.minimum_token_probability)
        fractions = [sum(supported(c) for c in word) / len(word) for word in words]
        spoken = [c for word in words for c in word]
        overall = sum(supported(c) for c in spoken) / len(spoken)
        margins = (mean_log_margin, final_log_margin)
        return (
            overall >= self.minimum_supported_fraction
            and supported(spoken[-1])
            and all(fraction >= self.minimum_word_supported_fraction
                    # Reduced short words can have weak CTC scores in fluent speech.
                    # Global support and the final-word gate still apply to them.
                    for word, fraction in zip(words, fractions)
                    if len(word) >= self.word_support_minimum_length)
            and fractions[-1] >= self.minimum_suffix_supported_fraction
            and all(margin is None or (math.isfinite(margin)
                                      and margin >= self.minimum_mean_log_margin)
                    for margin in margins)
        )


def ctc_end_boundary(log_probs, times, end_time, blank_id, separator_id, *,
                     minimum_probability=0.9, context_seconds=0.16):
    """Require a confident delimiter before any further decoded letter.

    This uses the same causal emissions as alignment, with no second inference.
    A later delimiter after extra letters must not turn a word prefix into an end.
    """
    if separator_id is None or not len(log_probs):
        return False
    after = log_probs[(times >= end_time - 1e-6)
                      & (times <= end_time + context_seconds + 1e-6)]
    if not len(after):
        return False
    tokens = after.argmax(axis=1)
    speech = np.flatnonzero(tokens != blank_id)
    if not len(speech):
        return False
    first = int(speech[0])
    return (int(tokens[first]) == separator_id
            and float(np.exp(after[first, separator_id])) >= minimum_probability)


def ctc_mean_log_margin(log_probs, targets, characters, times):
    """Compare aligned targets with their strongest competing token, per symbol."""
    if not len(log_probs) or log_probs.shape[1] < 2:
        return float("-inf")
    winners = log_probs.argmax(axis=1)
    highest = log_probs.max(axis=1)
    second = np.partition(log_probs, -2, axis=1)[:, -2]
    margins = []
    for character in characters:
        if not any(c.isalnum() for c in character.character):
            continue
        frames = (times >= character.start_time - 1e-6) & (times < character.end_time - 1e-6)
        if not frames.any():
            return float("-inf")
        target = targets[character.index]
        alternatives = np.where(winners == target, second, highest)
        margins.append(float(np.mean(log_probs[frames, target] - alternatives[frames])))
    return float(np.mean(margins)) if margins else float("-inf")
