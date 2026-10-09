"""Conservative Unicode text-offset projection for alignment spans."""

from __future__ import annotations

from dataclasses import replace
import unicodedata
from typing import Any, Iterable, Optional


def _normalise_with_indices(text: str) -> tuple[str, list[int]]:
    output: list[str] = []
    indices: list[int] = []
    for index, character in enumerate(text):
        normalised = unicodedata.normalize("NFKC", character).casefold()
        for item in normalised:
            if item.isspace():
                if output and output[-1] != " ":
                    output.append(" ")
                    indices.append(index)
            else:
                output.append(item)
                indices.append(index)
    while output and output[-1] == " ":
        output.pop()
        indices.pop()
    return "".join(output), indices


def _span_source(span: Any, spoken_text: Optional[str]) -> str:
    start = getattr(span, "text_start", None)
    end = getattr(span, "text_end", None)
    if (
        spoken_text is not None
        and isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start <= end <= len(spoken_text)
    ):
        source = spoken_text[start:end]
        if source:
            return source
    return str(getattr(span, "text", ""))


def project_text_offsets(
    spans: Iterable[Any],
    original_text: str,
    *,
    spoken_text: Optional[str] = None,
) -> list[Any]:
    """Project normalized or spoken spans to original Unicode codepoint offsets.

    Matching is sequential and exact after NFKC, casefold, and whitespace
    normalization. This preserves repeated-word order. If a normalized
    expansion cannot be matched without guessing, both offsets remain None.
    """

    if not isinstance(original_text, str):
        raise TypeError("original_text must be a string")
    if spoken_text is not None and not isinstance(spoken_text, str):
        raise TypeError("spoken_text must be a string or None")
    original_normalized, original_indices = _normalise_with_indices(original_text)
    cursor = 0
    projected: list[Any] = []
    for span in spans:
        source = _span_source(span, spoken_text)
        target, _ = _normalise_with_indices(source)
        start = end = None
        if target:
            match_start = original_normalized.find(target, cursor)
            if match_start >= 0:
                match_end = match_start + len(target)
                start = original_indices[match_start]
                end = original_indices[match_end - 1] + 1
                cursor = match_end
        try:
            projected.append(replace(span, text_start=start, text_end=end))
        except TypeError as exc:
            raise TypeError("spans must contain dataclass alignment spans") from exc
    return projected


def map_text_offsets(
    spans: Iterable[Any],
    original_text: str,
    *,
    spoken_text: Optional[str] = None,
) -> list[Any]:
    return project_text_offsets(spans, original_text, spoken_text=spoken_text)


def map_alignment_offsets(
    spans: Iterable[Any],
    original_text: str,
    *,
    spoken_text: Optional[str] = None,
) -> list[Any]:
    return project_text_offsets(spans, original_text, spoken_text=spoken_text)


__all__ = [
    "map_alignment_offsets",
    "map_text_offsets",
    "project_text_offsets",
]
