"""Explicit gaps shared by local and WebSocket language streams."""
from __future__ import annotations

import math

END_PUNCTUATION = ".!?。！？…¡¿"
MID_PUNCTUATION = ";:,\n()[]{}-“”„”—/|《》，、"
_PREFIX_PUNCTUATION = ".!?。！？…;:,，、)]}”’»"


def prefix_source_ranges(source, spoken, ranges=None):
    """Keep UTF-16 source offsets when closing punctuation moves to a gap."""
    prefix = source[:len(source) - len(spoken)]
    prefix_offset = len(prefix.encode("utf-16-le")) // 2
    if not prefix_offset:
        return ranges
    if ranges is None:
        ranges, offset = [], 0
        for char in spoken:
            end = offset + len(char.encode("utf-16-le")) // 2
            ranges.append([offset, end])
            offset = end
    return [[a + prefix_offset, b + prefix_offset] for a, b in ranges]


class SpeechBoundaryPauses:
    """Track emitted silence, so a language switch only adds a missing gap.

    No lookahead or audio buffering is needed: trailing punctuation is handled
    immediately, and punctuation arriving after a language tag tops up that gap.
    """

    def __init__(self, comma=0.15, sentence=0.30, transition=0.08, default=0.0):
        values = (comma, sentence, transition, default)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("silence durations must be finite and non-negative")
        self.comma, self.sentence, self.transition, self.default = values
        self.language = None
        self.emitted = 0.0

    def trailing(self, text):
        text = text.rstrip()
        if text and text[-1] in END_PUNCTUATION:
            return self.sentence
        if text and text[-1] in MID_PUNCTUATION:
            return self.comma
        return self.default

    def before(self, text, language):
        """Compute a gap; call explicit_pause only after that PCM is emitted."""
        if self.language is None:
            return text, 0.0
        changed = language != self.language
        punctuation_only = not any(char.isalnum() for char in text)
        desired = self.transition if changed else 0.0
        if changed or punctuation_only:
            end = 0
            for char in text:
                if char.isspace():
                    end += 1
                elif char in _PREFIX_PUNCTUATION:
                    desired = max(desired, self.trailing(char))
                    end += 1
                else:
                    break
            if end < len(text) and text[end].isalnum() and end and not text[end - 1].isspace():
                # .NET and similar tokens own their attached punctuation.
                desired = self.transition if changed else 0.0
            else:
                text = text[end:]
        extra = max(0.0, desired - self.emitted)
        return text, extra

    def after(self, text, language):
        self.language = language
        self.emitted = 0.0
        return self.trailing(text)

    def explicit_pause(self, duration):
        self.emitted += duration
