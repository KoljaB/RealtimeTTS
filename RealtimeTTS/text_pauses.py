"""Choose the pause that follows one synthesized fragment."""

_TRAILING_CLOSERS = frozenset("\"'”’»)]}）】」』〉》")


def pause_delimiter(text: str, sentence_delimiters: str) -> str:
    """Return the character that selects the pause after *text*.

    A sentence delimiter immediately before closing quotes or brackets wins;
    otherwise the fragment's original final character keeps its existing class.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    original = stripped[-1]
    while stripped and stripped[-1] in _TRAILING_CLOSERS:
        stripped = stripped[:-1].rstrip()
    if stripped and stripped[-1] in sentence_delimiters:
        return stripped[-1]
    return original
