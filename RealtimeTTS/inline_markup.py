"""Bounded incremental parser for inline language, speed, voice, and pause markup.

The parser deliberately has no engine or playback dependencies. It turns a
character stream into text and control actions while retaining scope state on
the parser instance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from .audio_tempo import validate_speed


Action = Tuple[str, Any]


@dataclass
class _VoiceState:
    tag: Optional[str]
    voice: Any
    known: bool


@dataclass
class _Scope:
    kind: str
    name: str
    previous_speed: Optional[float] = None
    previous_voice: Optional[_VoiceState] = None
    previous_language: Optional[str] = None


@dataclass
class _ParserState:
    speed: float
    voice: _VoiceState
    scopes: List[_Scope]
    language: str
    language_depth: int = 0


class InlineMarkupParser:
    """Parse inline controls without buffering the source body.

    voice_tags and pause_tags are mappings using the same tag names accepted by
    the legacy bracket syntax. initial_voice is a (registered_tag, voice_object)
    pair when a known baseline is available. An unknown baseline is retained as
    unknown state; restoring it therefore emits no voice action.
    """

    MAX_TAG_LENGTH = 256
    MAX_LANGUAGE_DEPTH = 64
    LANGUAGE_CODES = frozenset(("en", "de", "es", "pt", "fr", "it", "ru", "zh", "ja", "ko"))
    LANGUAGE_NAMES = {"english": "en", "german": "de", "spanish": "es",
                      "portuguese": "pt", "french": "fr", "italian": "it",
                      "russian": "ru", "chinese": "zh", "japanese": "ja", "korean": "ko"}
    _LANGUAGE_SPEEDS = {"slow": 0.75, "normal": 1.0, "fast": 1.2}
    _SPEED_NAMES = frozenset(("speed", "sr", "rate"))
    _VOICE_NAMES = frozenset(("voice", "style", "emotion"))
    _NUMERIC = re.compile(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?"
    )
    _NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_:-]*")
    _ATTRIBUTE = re.compile(
        r"""([A-Za-z_][A-Za-z0-9_:-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>]+))"""
    )

    def __init__(
        self,
        default_speed: float,
        *,
        speed_tags: bool = False,
        language_tags: bool = False,
        default_language: str = "en",
        language_voices: Optional[Mapping[str, Any]] = None,
        voice_tags: Optional[Mapping[str, Any]] = None,
        pause_tags: Optional[Mapping[str, float]] = None,
        initial_voice: Optional[Tuple[Optional[str], Any]] = None,
        delimiters: Tuple[str, str] = ("[", "]"),
    ) -> None:
        self.default_speed = validate_speed(default_speed)
        self.language_tags = bool(language_tags)
        self.speed_tags = bool(speed_tags or language_tags)
        self.default_language = str(default_language).strip().lower()
        if not self.default_language:
            raise ValueError("default_language must not be empty")
        self.language_voices = {}
        for code, voice in (language_voices or {}).items():
            code = self.language_code(code)
            if code not in self.LANGUAGE_CODES:
                raise ValueError(f"Unsupported language voice route: {code!r}")
            self.language_voices[code] = voice

        self.voice_tags: Dict[str, Any] = {}
        for tag, voice in (voice_tags or {}).items():
            key = self._tag_key(tag, "voice")
            self.voice_tags[key] = voice

        self.pause_tags: Dict[str, float] = {}
        for tag, duration in (pause_tags or {}).items():
            key = self._tag_key(tag, "pause")
            self.pause_tags[key] = self._validate_pause(duration)

        if (
            not isinstance(delimiters, (tuple, list))
            or len(delimiters) != 2
            or not isinstance(delimiters[0], str)
            or not isinstance(delimiters[1], str)
            or not delimiters[0]
            or not delimiters[1]
        ):
            raise ValueError("delimiters must be a pair of non-empty strings")
        self.delimiters = (delimiters[0], delimiters[1])

        if initial_voice is not None:
            if (
                not isinstance(initial_voice, (tuple, list))
                or len(initial_voice) != 2
            ):
                raise ValueError("initial_voice must be a (tag, voice) pair")
            raw_initial_tag = initial_voice[0]
            initial_tag = (
                None
                if raw_initial_tag is None
                else self._tag_key(raw_initial_tag, "initial voice")
            )
            self.initial_voice = (initial_tag, initial_voice[1])
        else:
            self.initial_voice = None

        # A configured pair wins when it shares an opener with a built-in.
        self._delimiter_pairs: List[Tuple[str, str]] = [self.delimiters]
        for builtin in (("[", "]"), ("<", ">")):
            if builtin not in self._delimiter_pairs:
                self._delimiter_pairs.append(builtin)
        self._starts = tuple(
            sorted({pair[0] for pair in self._delimiter_pairs}, key=len, reverse=True)
        )
        self._enabled = bool(self.language_tags or self.speed_tags or self.voice_tags or self.pause_tags)
        self._legacy_state = self._new_state()

    @staticmethod
    def _tag_key(value: Any, label: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{label} tag must be a string")
        key = value.strip()
        if not key:
            raise ValueError(f"{label} tag must not be empty")
        return key

    @staticmethod
    def _validate_pause(value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError("pause duration must be a finite non-negative float")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "pause duration must be a finite non-negative float"
            ) from exc
        if not math.isfinite(result) or result < 0:
            raise ValueError("pause duration must be a finite non-negative float")
        return result

    def _new_state(self) -> _ParserState:
        if self.initial_voice is None:
            voice = _VoiceState(None, None, False)
        else:
            tag, value = self.initial_voice
            if tag is None:
                voice = _VoiceState(None, value, True)
            else:
                resolved = self._lookup_voice(tag)
                if resolved is None:
                    voice = _VoiceState(tag, value, False)
                else:
                    canonical, _registered = resolved
                    voice = _VoiceState(canonical, value, True)
        return _ParserState(self.default_speed, voice, [], self.default_language)

    def _lookup(
        self, mapping: Mapping[str, Any], query: str
    ) -> Optional[Tuple[str, Any]]:
        query = query.strip()
        if not query:
            return None
        if query in mapping:
            return query, mapping[query]
        folded = query.casefold()
        matches = [(key, value) for key, value in mapping.items()
                   if key.casefold() == folded]
        if len(matches) != 1:
            return None
        return matches[0]

    def _lookup_voice(self, query: str) -> Optional[Tuple[str, Any]]:
        return self._lookup(self.voice_tags, query)

    def _lookup_pause(self, query: str) -> Optional[Tuple[str, float]]:
        return self._lookup(self.pause_tags, query)  # type: ignore[return-value]

    @classmethod
    def _split_name(cls, content: str) -> Optional[Tuple[str, str]]:
        match = cls._NAME.match(content)
        if match is None:
            return None
        return match.group(0), content[match.end():]

    @classmethod
    def _scalar(cls, value: str) -> Optional[str]:
        value = value.strip()
        if not value:
            return None
        if value[0] in "\"'":
            quote = value[0]
            if len(value) < 2 or value[-1] != quote:
                return None
            inner = value[1:-1]
            if quote in inner:
                return None
            return inner
        if any(char.isspace() for char in value):
            return None
        return value

    @classmethod
    def _attributes(cls, value: str) -> Optional[Dict[str, str]]:
        attributes: Dict[str, str] = {}
        cursor = 0
        while cursor < len(value):
            while cursor < len(value) and value[cursor].isspace():
                cursor += 1
            if cursor == len(value):
                break
            match = cls._ATTRIBUTE.match(value, cursor)
            if match is None:
                return None
            key = match.group(1).casefold()
            if key in attributes:
                return None
            attributes[key] = (
                match.group(2)
                if match.group(2) is not None
                else match.group(3)
                if match.group(3) is not None
                else match.group(4)
            )
            cursor = match.end()
        return attributes

    @classmethod
    def _number(cls, value: str, *, percent: bool = False) -> Optional[float]:
        value = value.strip()
        if cls._NUMERIC.fullmatch(value) is None:
            return None
        is_percent = value.endswith("%")
        if is_percent and not percent:
            return None
        raw = value[:-1] if is_percent else value
        try:
            result = float(raw)
        except ValueError:
            return None
        if is_percent:
            result /= 100.0
        return result

    def _speed_value(self, raw: str) -> Optional[float]:
        raw = raw.strip()
        if raw.casefold() == "reset":
            return self.default_speed
        value = self._number(raw, percent=True)
        if value is None:
            return None
        try:
            return validate_speed(value)
        except ValueError:
            return None

    def _parse_speed(self, content: str) -> Optional[Tuple[str, float]]:
        split = self._split_name(content)
        if split is None:
            return None
        name, tail = split
        name_folded = name.casefold()
        if not self.speed_tags or name_folded not in self._SPEED_NAMES:
            return None

        tail = tail.strip()
        raw_value: Optional[str] = None
        if tail.startswith("=") or tail.startswith(":"):
            raw_value = self._scalar(tail[1:])
        elif tail:
            attributes = self._attributes(tail)
            if attributes is not None:
                values = [
                    attributes[key]
                    for key in ("factor", "rate", "value")
                    if key in attributes
                ]
                if len(values) != 1 or any(
                    key not in ("factor", "rate", "value") for key in attributes
                ):
                    return None
                raw_value = values[0]
            else:
                raw_value = self._scalar(tail)
        if raw_value is None:
            return None
        value = self._speed_value(raw_value)
        if value is None:
            return None
        return name_folded, value

    def _parse_voice(self, content: str) -> Optional[Tuple[str, Any]]:
        content = content.strip()
        direct = self._lookup_voice(content)
        if direct is not None:
            return direct
        split = self._split_name(content)
        if split is None:
            return None
        name, tail = split
        if not tail.strip():
            return self._lookup_voice(name)

        name_folded = name.casefold()
        if name_folded not in self._VOICE_NAMES:
            return None

        tail = tail.strip()
        query: Optional[str] = None
        if tail.startswith("="):
            query = self._scalar(tail[1:])
        else:
            attributes = self._attributes(tail)
            if attributes is None:
                return None
            allowed = {"name", "style", "value"}
            if any(key not in allowed for key in attributes):
                return None
            values = [attributes[key] for key in ("name", "style", "value")
                      if key in attributes]
            if len(values) != 1:
                return None
            query = values[0]
        if query is None:
            return None
        return self._lookup_voice(query)

    def _parse_pause(
        self, content: str
    ) -> Optional[Tuple[str, float]]:
        if not self.pause_tags:
            return None
        content = content.strip()

        direct = self._lookup_pause(content)
        if direct is not None:
            return direct

        # Preserve arbitrary registered names, including spaces and Unicode,
        # for the legacy override forms as well as for bare tags.
        for registered in sorted(self.pause_tags, key=len, reverse=True):
            if len(content) <= len(registered):
                continue
            if content[:len(registered)].casefold() != registered.casefold():
                continue
            remainder = content[len(registered):]
            if not remainder or (
                not remainder[0].isspace() and remainder[0] not in "=:"
            ):
                continue
            raw_duration = remainder.strip()
            if raw_duration.startswith("=") or raw_duration.startswith(":"):
                raw_duration = raw_duration[1:].strip()
            elif raw_duration:
                attributes = self._attributes(raw_duration)
                if attributes is not None and all(
                    key in ("duration", "value") for key in attributes
                ):
                    values = [attributes[key] for key in ("duration", "value")
                              if key in attributes]
                    raw_duration = values[0] if len(values) == 1 else ""
                else:
                    raw_duration = self._scalar(raw_duration) or ""
            duration = self._number(raw_duration)
            resolved = self._lookup_pause(registered)
            if resolved is None or duration is None:
                continue
            try:
                return resolved[0], self._validate_pause(duration)
            except ValueError:
                continue

        split = self._split_name(content)
        if split is None:
            return None
        name, tail = split
        tail = tail.strip()
        raw_duration: Optional[str] = None

        if not tail:
            return self._lookup_pause(name)
        if tail.startswith("=") or tail.startswith(":"):
            raw_duration = self._scalar(tail[1:])
        else:
            attributes = self._attributes(tail)
            if attributes is not None and all(
                key in ("duration", "value") for key in attributes
            ):
                values = [attributes[key] for key in ("duration", "value")
                          if key in attributes]
                if len(values) == 1:
                    raw_duration = values[0]
            if raw_duration is None:
                raw_duration = self._scalar(tail)

        resolved = self._lookup_pause(name)
        if resolved is None or raw_duration is None:
            return None
        duration = self._number(raw_duration)
        if duration is None:
            return None
        try:
            duration = self._validate_pause(duration)
        except ValueError:
            return None
        return resolved[0], duration

    @classmethod
    def language_code(cls, language: str) -> str:
        value = str(language).strip().lower()
        return cls.LANGUAGE_NAMES.get(value, value)

    def _parse_language(self, content: str) -> Optional[Tuple[str, float]]:
        if not self.language_tags:
            return None
        split = self._split_name(content)
        if split is None:
            return None
        name, tail = split
        name = name.casefold()
        tail = tail.strip()
        speed = self.default_speed
        if name == "lang":
            attributes = self._attributes(tail)
            if attributes is None or "code" not in attributes or set(attributes) - {"code", "speed"}:
                return None
            code = attributes["code"].casefold()
            raw_speed = attributes.get("speed")
        elif name in self.LANGUAGE_CODES:
            code = name
            raw_speed = tail or None
            if tail and tail.casefold() not in self._LANGUAGE_SPEEDS:
                attributes = self._attributes(tail)
                if attributes is None or set(attributes) != {"speed"}:
                    return None
                raw_speed = attributes["speed"]
        else:
            return None
        if code not in self.LANGUAGE_CODES:
            return None
        if raw_speed is not None:
            speed = self._LANGUAGE_SPEEDS.get(raw_speed.casefold())
            if speed is None:
                speed = self._speed_value(raw_speed)
            if speed is None:
                return None
        return code, speed

    def _open_language(self, state: _ParserState, name: str, code: str,
                       speed: float, self_closing: bool) -> List[Action]:
        route = self.language_voices.get(code)
        if not self_closing:
            if state.language_depth >= self.MAX_LANGUAGE_DEPTH:
                raise ValueError("Too many nested speech language controls")
            state.scopes.append(_Scope("language", name,
                previous_speed=state.speed, previous_language=state.language,
                previous_voice=state.voice if route is not None else None))
            state.language_depth += 1
        actions = []
        if route is not None:
            state.voice = _VoiceState(code, route, True)
            actions.append(("voice", (code, route)))
        state.language = code
        actions.append(("language", code))
        if speed != state.speed:
            state.speed = speed
            actions.append(("speed", speed))
        return actions

    def _open_speed(
        self, state: _ParserState, name: str, value: float, self_closing: bool
    ) -> List[Action]:
        if not self_closing:
            state.scopes.append(_Scope("speed", name, previous_speed=state.speed))
        state.speed = value
        return [("speed", value)]

    def _open_voice(
        self,
        state: _ParserState,
        tag: str,
        voice: Any,
        self_closing: bool,
    ) -> List[Action]:
        if not self_closing:
            state.scopes.append(
                _Scope("voice", tag, previous_voice=state.voice)
            )
        state.voice = _VoiceState(tag, voice, True)
        return [("voice", (tag, voice))]

    def _close_scope(
        self, name: str, state: _ParserState
    ) -> Optional[List[Action]]:
        name = name.strip()
        if not name:
            return None
        kind: Optional[str] = None
        canonical_name: Optional[str] = None

        if self.language_tags and (name.casefold() == "lang" or name.casefold() in self.LANGUAGE_CODES):
            kind = "language"
            canonical_name = name.casefold()
        elif name.casefold() in self._SPEED_NAMES and self.speed_tags:
            kind = "speed"
            canonical_name = name.casefold()
        elif name.casefold() in self._VOICE_NAMES and self.voice_tags:
            kind = "voice"
        else:
            resolved = self._lookup_voice(name)
            if resolved is not None:
                kind = "voice"
                canonical_name = resolved[0]
            else:
                return None

        match_index: Optional[int] = None
        for index in range(len(state.scopes) - 1, -1, -1):
            scope = state.scopes[index]
            if scope.kind != kind:
                continue
            if kind in ("voice", "language") and canonical_name is not None:
                if scope.name.casefold() != canonical_name.casefold():
                    continue
            match_index = index
            break

        if match_index is None:
            return None if kind == "language" else []

        actions: List[Action] = []
        while len(state.scopes) > match_index:
            scope = state.scopes.pop()
            if scope.kind == "speed":
                # Every speed scope carries a concrete previous speed.
                state.speed = scope.previous_speed  # type: ignore[assignment]
                actions.append(("speed", state.speed))
            elif scope.kind == "language":
                state.language_depth -= 1
                if scope.previous_voice is not None:
                    state.voice = scope.previous_voice
                    if state.voice.known:
                        actions.append(("voice", (state.voice.tag, state.voice.voice)))
                state.language = scope.previous_language  # type: ignore[assignment]
                actions.append(("language", state.language))
                if state.speed != scope.previous_speed:
                    state.speed = scope.previous_speed  # type: ignore[assignment]
                    actions.append(("speed", state.speed))
            else:
                previous = scope.previous_voice
                if previous is None:
                    continue
                state.voice = previous
                if previous.known:
                    actions.append(("voice", (previous.tag, previous.voice)))
        return actions

    def _resolve_content(
        self,
        content: str,
        state: _ParserState,
        self_closing: bool = False,
    ) -> Optional[List[Action]]:
        content = content.strip()
        if not content:
            return None

        if not content.startswith("/") and content.endswith("/"):
            self_closing = True
            content = content[:-1].rstrip()
            if not content:
                return None

        if content.startswith("/"):
            return self._close_scope(content[1:].strip(), state)

        language = self._parse_language(content)
        if language is not None:
            name = content.split()[0].casefold()
            return self._open_language(state, name, language[0], language[1], self_closing)

        speed = self._parse_speed(content)
        if speed is not None:
            return self._open_speed(state, speed[0], speed[1], self_closing)

        voice = self._parse_voice(content)
        if voice is not None:
            return self._open_voice(state, voice[0], voice[1], self_closing)

        pause = self._parse_pause(content)
        if pause is not None:
            return [("pause", pause)]

        return None

    def resolve_tag(
        self, content: str, self_closing: bool = False
    ) -> Optional[List[Action]]:
        """Resolve one already-delimited tag using compatibility state.

        Unknown or invalid content returns None so callers can preserve its
        original literal spelling. Known unmatched closing tags return an empty
        action list and are intentionally consumed.
        """
        if not isinstance(content, str):
            return None
        return self._resolve_content(content, self._legacy_state, self_closing)

    @property
    def current_language(self) -> str:
        """Current logical language, including a scope continued by late feed."""
        return self._legacy_state.language

    def reset(self) -> None:
        """Reset state used by direct resolve_tag calls."""
        self._legacy_state = self._new_state()

    def _start_in_probe(
        self, probe: str
    ) -> Optional[Tuple[int, str]]:
        matches: List[Tuple[int, str]] = []
        for index in range(len(probe)):
            for start in self._starts:
                if probe.startswith(start, index):
                    matches.append((index, start))
        if not matches:
            return None
        earliest = min(item[0] for item in matches)
        same_position = [item[1] for item in matches if item[0] == earliest]
        return earliest, max(same_position, key=len)

    def _possible_start_suffix(self, probe: str) -> int:
        longest = 0
        for start in self._starts:
            for size in range(1, min(len(start), len(probe)) + 1):
                if probe.endswith(start[:size]):
                    longest = max(longest, size)
        return longest

    @staticmethod
    def _is_escaped(value: str) -> bool:
        backslashes = 0
        for character in reversed(value[:-1]):
            if character != "\\":
                break
            backslashes += 1
        return bool(backslashes % 2)

    def iter_actions(
        self,
        source: Iterable[str],
        stopped: Callable[[], bool] = lambda: False,
    ) -> Iterator[Action]:
        """Yield actions incrementally, retaining at most one tag candidate."""
        # The state survives successive iter_actions calls until reset().
        state = self._legacy_state
        if not self._enabled:
            for chunk in source:
                for character in chunk:
                    if stopped():
                        return
                    yield ("text", character)
            return

        probe = ""
        candidate = ""
        active_start = ""
        active_end = ""
        quote: Optional[str] = None
        in_tag = False

        for chunk in source:
            for character in chunk:
                if stopped():
                    return
                if not in_tag:
                    probe += character
                    found = self._start_in_probe(probe)
                    if found is not None:
                        index, start = found
                        # If a shorter start is also a prefix of a longer
                        # delimiter, wait for one more character.
                        suffix = probe[index:]
                        if (
                            index + len(start) == len(probe)
                            and any(
                                other != start
                                and other.startswith(suffix)
                                and len(other) > len(start)
                                for other in self._starts
                            )
                        ):
                            continue
                        for literal in probe[:index]:
                            yield ("text", literal)
                        tail = probe[index + len(start):]
                        probe = ""
                        candidate = start
                        active_start = start
                        active_end = next(
                            end
                            for begin, end in self._delimiter_pairs
                            if begin == start
                        )
                        quote = None
                        in_tag = True
                        # A tail can only occur with overlapping delimiters,
                        # but processing it here preserves every character.
                        for tail_character in tail:
                            candidate += tail_character
                    else:
                        keep = self._possible_start_suffix(probe)
                        if keep < len(probe):
                            for literal in probe[:-keep] if keep else probe:
                                yield ("text", literal)
                            probe = probe[-keep:] if keep else ""
                    continue

                candidate += character
                if quote is not None:
                    if character == quote and not self._is_escaped(candidate):
                        quote = None
                elif character in ("'", '"'):
                    quote = character

                if len(candidate) > self.MAX_TAG_LENGTH:
                    yield ("text", candidate)
                    candidate = ""
                    active_start = ""
                    active_end = ""
                    quote = None
                    in_tag = False
                elif (
                    quote is None
                    and len(candidate) > len(active_start)
                    and candidate.endswith(active_end)
                ):
                    raw = candidate
                    content = raw[len(active_start):-len(active_end)]
                    closing = content.rstrip().endswith("/")
                    actions = self._resolve_content(content, state, closing)
                    if actions is None:
                        yield ("text", raw)
                    else:
                        for action in actions:
                            yield action
                    candidate = ""
                    active_start = ""
                    active_end = ""
                    quote = None
                    in_tag = False

        if stopped():
            return
        if in_tag and candidate:
            yield ("text", candidate)
        for literal in probe:
            if stopped():
                return
            yield ("text", literal)


__all__ = ["InlineMarkupParser"]
