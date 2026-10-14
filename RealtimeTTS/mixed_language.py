"""Opt-in fastText repair for missing tutor language controls.

Runs only on fragments already released by stream2sentence. Explicit language
controls always take precedence. No stream lookahead or additional LLM request.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_WORD = re.compile(r"[^\W\d_]+(?:[’'-][^\W\d_]+)*", re.UNICODE)
_BREAK = re.compile(r"""[.!?;:\n。！？"“”„«»‘’']""")
_QUOTED = re.compile(r'"([^"\n]*)"|“([^”\n]*)”|„([^“\n]*)“|«([^»\n]*)»|‘([^’\n]*)’|(?<!\w)\'([^\'\n]+)\'(?!\w)')
# Closed grammatical classes, not a translation dictionary. A shared article
# belongs with a confidently identified noun; it is not a foreign anchor alone.
# Spanish "es" is also a German pronoun, so it cannot anchor a repair window.
_PARTICLES = {
    "spanish": frozenset("el la los las un una unos unas a al de del en con por para sin es".split()),
    "portuguese": frozenset("o a os as um uma uns umas à às ao aos de do da dos das em no na nos nas por pelo pela pelos pelas para com sem".split()),
}


@dataclass(frozen=True)
class LanguageSpan:
    text: str
    language: str | None = None


def split_mixed_language(text: str, target: str, detector: Any) -> list[LanguageSpan]:
    """Preserve every character, labelling supported target-language islands.

    The target is known from tutor mode. Comparing its probability with the
    explanation language avoids rejecting Portuguese solely because a related
    language scored slightly higher. Weak isolated words stay unmodified.
    """
    words = list(_WORD.finditer(text))
    if target not in _PARTICLES or not words or len(text) > 2048 or len(words) > 128:
        return [LanguageSpan(text)]
    particles = _PARTICLES[target]
    quoted_ranges = set()
    for quote in _QUOTED.finditer(text):
        indexes = [i for i, word in enumerate(words)
                   if quote.start() <= word.start() and word.end() <= quote.end()]
        if indexes:
            quoted_ranges.add((indexes[0], indexes[-1] + 1))
    whole = detector.detect(" ".join(text.split()))
    base = (whole.candidate_language
            if whole.candidate_language in ("english", "german")
            and float(whole.confidence or 0) >= .55
            else detector.fallback_language)
    cache = {}

    def scores(start, end):
        key = (start, end)
        if key not in cache:
            cache[key] = detector.language_probabilities(text[words[start].start():words[end-1].end()])
        value = cache[key]
        foreign, surrounding = value.get(target, 0.0), value.get(base, 0.0)
        return foreign, foreign / max(1e-9, foreign + surrounding), foreign - surrounding

    def grammatical(start, end):
        return all(words[i].group().casefold() in particles for i in range(start, end))

    def supported(absolute, relative, margin):
        return (absolute >= .45 and margin >= .25) or (absolute >= .12 and relative >= .95)

    candidates = []
    for start in range(len(words)):
        for end in range(start+1, min(start+6, len(words))+1):
            if end > start+1 and _BREAK.search(text[words[end-2].end():words[end-1].start()]):
                break
            if grammatical(start, end) or words[end-1].group().casefold() in particles:
                continue
            absolute, relative, margin = scores(start, end)
            if end == start+1:
                if absolute < .98 or relative < .98 or len(words[start].group()) < 3:
                    continue
            elif not supported(absolute, relative, margin):
                continue
            elif (start, end) not in quoted_ranges and not any(
                words[i].group().casefold() not in particles
                and scores(i, i + 1)[0] >= .05
                and scores(i, i + 1)[1] >= .50 for i in range(start, end)
            ):
                # Shared function words must not turn surrounding explanation
                # words into a foreign phrase (German "Hör es dir an").
                continue
            candidates.append(((start, end) in quoted_ranges, absolute, start, end))
    occupied = set()
    selected = []
    for _, _, start, end in sorted(candidates, reverse=True):
        if any(i in occupied for i in range(start, end)):
            continue
        # Keep target articles/prepositions attached, using phrase evidence,
        # even if their isolated spelling is shared with English.
        while start > 0 and start-1 not in occupied and grammatical(start-1, start):
            if _BREAK.search(text[words[start-1].end():words[start].start()]):
                break
            absolute, relative, margin = scores(start-1, end)
            if not supported(absolute, relative, margin):
                break
            start -= 1
        occupied.update(range(start, end))
        selected.append((start, end))
    if not selected:
        return [LanguageSpan(text)]
    joined = []
    for start, end in sorted(selected):
        if joined:
            left, right = joined[-1]
            gap = text[words[right-1].end():words[start].start()]
            if not _BREAK.search(gap) and (start == right or grammatical(right, start)):
                absolute, relative, margin = scores(left, end)
                if supported(absolute, relative, margin):
                    joined[-1] = (left, end)
                    continue
        joined.append((start, end))
    spans = []
    cursor = 0
    for start, end in joined:
        begin, finish = words[start].start(), words[end-1].end()
        if begin > cursor:
            spans.append(LanguageSpan(text[cursor:begin]))
        spans.append(LanguageSpan(text[begin:finish], target))
        cursor = finish
    if cursor < len(text):
        spans.append(LanguageSpan(text[cursor:]))
    return spans
