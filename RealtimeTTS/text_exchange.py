"""Case-sensitive, whole-word pronunciation exchanges for speech requests."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from .language_router import normalize_qwen_language


def _table(value, label):
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a string-to-string object")
    result = {}
    for key, replacement in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{label} keys must be non-empty strings")
        if not isinstance(replacement, str) or not replacement.strip():
            raise ValueError(f"{label} values must be non-empty strings")
        result[key] = replacement
    return result


class TextExchange:
    """Compile once per HTTP request or WebSocket config, never per text chunk."""

    def __init__(self, exchange=None, exchange_by_language=None):
        common = _table({} if exchange is None else exchange, "exchange")
        per_language = {} if exchange_by_language is None else exchange_by_language
        if not isinstance(per_language, Mapping):
            raise ValueError("exchange_by_language must be an object of exchange objects")
        tables = {"auto": common}
        for language, values in per_language.items():
            language = normalize_qwen_language(language)
            if language in tables:
                raise ValueError(f"duplicate exchange language: {language}")
            tables[language] = {**common, **_table(values, "exchange_by_language")}
        self._rules = {}
        for language, table in tables.items():
            # Longest match wins; replacements are literal and never run through again.
            keys = sorted(table, key=len, reverse=True)
            pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, keys)) + r")(?!\w)") if keys else None
            self._rules[language] = (pattern, table)

    def apply(self, text, language):
        language = normalize_qwen_language(language, allow_auto=True)
        pattern, table = self._rules.get(language, self._rules["auto"])
        if pattern is None:
            return text, None
        matches = [m for m in pattern.finditer(text)
                   if table[m.group()] != m.group()
                   and not (m.start() and unicodedata.category(text[m.start() - 1]).startswith("M"))
                   and not (m.end() < len(text) and unicodedata.category(text[m.end()]).startswith("M"))]
        if not matches:
            return text, None
        # Exact original UTF-16 ranges let clients keep displaying the source spelling.
        original_ranges, offset = [], 0
        for character in text:
            end = offset + len(character.encode("utf-16-le")) // 2
            original_ranges.append([offset, end])
            offset = end
        pieces, ranges, cursor = [], [], 0
        for match in matches:
            pieces.append(text[cursor:match.start()])
            ranges.extend(original_ranges[cursor:match.start()])
            replacement = table[match.group()]
            pieces.append(replacement)
            source = original_ranges[match.start():match.end()]
            for index in range(len(replacement)):
                first = index * len(source) // len(replacement)
                last = ((index + 1) * len(source) - 1) // len(replacement)
                ranges.append([source[first][0], source[last][1]])
            cursor = match.end()
        pieces.append(text[cursor:])
        ranges.extend(original_ranges[cursor:])
        return "".join(pieces), ranges
