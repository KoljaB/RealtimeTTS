"""Language-aware routing facade for PCM streaming engines."""

from __future__ import annotations

import math
import queue as queue_module
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .base_engine import BaseEngine

SUPPORTED_STREAM_INFO = (8, 1, 24000)

_LANGUAGE_ALIASES = {
    "en": "en", "eng": "en", "english": "en", "en-us": "en", "en-gb": "en",
    "zh": "zh", "cmn": "zh", "chinese": "zh", "mandarin": "zh", "zh-cn": "zh",
    "zh-hans": "zh", "ja": "ja", "jpn": "ja", "japanese": "ja", "jp": "ja",
    "ko": "ko", "kor": "ko", "korean": "ko", "kr": "ko",
    "de": "de", "deu": "de", "german": "de", "deutsch": "de", "de-de": "de",
    "fr": "fr", "fra": "fr", "french": "fr", "fr-fr": "fr",
    "es": "es", "spa": "es", "spanish": "es", "es-es": "es",
    "it": "it", "ita": "it", "italian": "it", "it-it": "it",
    "pt": "pt", "por": "pt", "portuguese": "pt", "pt-br": "pt", "pt-pt": "pt",
    "ru": "ru", "rus": "ru", "russian": "ru", "ru-ru": "ru",
}
_CANONICAL_LANGUAGE_NAMES = {
    "en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean",
    "de": "german", "fr": "french", "es": "spanish", "it": "italian",
    "pt": "portuguese", "ru": "russian",
}

_ALL_CONTEXTS = object()
_CURRENT_CONTEXT = object()


def normalize_language(language: str, *, allow_auto: bool = False) -> str:
    if not isinstance(language, str):
        raise TypeError("language must be a string")
    value = language.strip().lower().replace("_", "-")
    if value == "auto":
        if allow_auto:
            return value
        raise ValueError("auto is not a concrete routing language")
    if value == "*":
        return value
    try:
        return _LANGUAGE_ALIASES[value]
    except KeyError as exc:
        raise ValueError(f"unsupported routing language: {language!r}") from exc


@dataclass(frozen=True)
class LanguageRoute:
    engine: Any
    voice: Any = None
    voices: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.engine is None:
            raise ValueError("LanguageRoute.engine cannot be None")
        if self.voices is not None:
            if not isinstance(self.voices, Mapping):
                raise TypeError("LanguageRoute.voices must be a mapping")
            object.__setattr__(self, "voices", MappingProxyType(dict(self.voices)))


@dataclass
class _Attempt:
    provider: Any
    generation: int
    cancelled: threading.Event = field(default_factory=threading.Event)
    first_pcm: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None
    thread: threading.Thread | None = None
    committed_items: int = 0
    pcm_bytes: int = 0
    router_audio_start: float = 0.0
    child_audio_start: float = 0.0
    stop_requested: bool = False
    alignment_fenced: threading.Event = field(default_factory=threading.Event)
    alignment_offset: float = 0.0
    alignment_success: bool = False
    timed_out: bool = False
    alignment_bridge: Any = None
    context: Any = None
    original_queue: Any = None
    original_timings: Any = None
    has_timings: bool = False
    queue_sink: Any = None
    timings_sink: Any = None


class _NoPcmFailure(RuntimeError):
    pass


class _AttemptCancelled(RuntimeError):
    pass


@dataclass
class _RoutingStreamContext:
    contexts: dict[int, Any] = field(default_factory=dict)
    closed: bool = False


class _FencedSink:
    """Queue-compatible sink that drops output from a stale attempt."""

    def __init__(
        self, owner, attempt, downstream, *, commit_pcm, drop_if_native_callback=False
    ):
        self._owner = owner
        self._attempt = attempt
        self._downstream = downstream
        self._commit_pcm = commit_pcm
        self._drop_if_native_callback = drop_if_native_callback

    @property
    def state(self):
        return self._attempt

    def _active(self):
        return (
            not self._attempt.cancelled.is_set()
            and self._owner._is_active_attempt(self._attempt)
        )

    def _put(self, item, block, timeout):
        if not self._active():
            return False
        if (
            self._drop_if_native_callback
            and getattr(self._owner, "provides_word_timings", False)
            and callable(getattr(self._owner, "on_alignment", None))
        ):
            return True
        if not block:
            self._downstream.put_nowait(item)
        elif timeout is not None:
            deadline = time.monotonic() + max(0.0, timeout)
            while self._active():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue_module.Full
                try:
                    self._downstream.put(
                        item, block=True, timeout=min(remaining, 0.1)
                    )
                    break
                except queue_module.Full:
                    continue
            else:
                return False
        else:
            while self._active():
                try:
                    self._downstream.put(item, block=True, timeout=0.1)
                    break
                except queue_module.Full:
                    continue
            else:
                return False
        # A completed put may already be audible even if cancellation raced it.
        # Always account for that commit so timeout fallback cannot replay it.
        if self._commit_pcm:
            try:
                size = len(item)
            except TypeError:
                size = int(getattr(item, "nbytes", 1))
            if size > 0:
                self._attempt.committed_items += 1
                self._attempt.pcm_bytes += size
                self._attempt.first_pcm.set()
        return True

    def put(self, item, block=True, timeout=None):
        self._put(item, block, timeout)

    def put_nowait(self, item):
        self._put(item, False, 0.0)

    def qsize(self):
        return self._downstream.qsize()

    def empty(self):
        return self._downstream.empty()

    def full(self):
        return self._downstream.full()

    def get(self, *args, **kwargs):
        return self._downstream.get(*args, **kwargs)

    def get_nowait(self):
        return self._downstream.get_nowait()


class _AlignmentBridge:
    """Per-attempt native callback with a fixed owner callback and offset."""

    def __init__(self, owner, provider, original, callback, attempt, offset):
        self.owner = owner
        self.provider = provider
        self.original = original
        self.callback = callback
        self.attempt = attempt
        self.offset = float(offset)
        self._lock = threading.Lock()
        self._closed = False

    def fence(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            attempt = self.attempt
            self.attempt = None
        if attempt is not None:
            attempt.alignment_fenced.set()
        self.owner._release_alignment_bridge(self)

    def __call__(self, event):
        with self._lock:
            if self._closed:
                return
            callback = self.callback
            original = self.original
        translated = self.owner._translate_alignment_event(event, self.offset)
        if callable(callback):
            try:
                callback(translated)
            except Exception:
                pass
        if callable(original) and original is not callback:
            try:
                original(event)
            except Exception:
                pass
        if isinstance(event, Mapping) and event.get("type") == "alignment_done":
            self.fence()


class LanguageRoutingEngine(BaseEngine):
    """Serialize provider synthesis and fallback only before first PCM."""

    language_engine_routing = True
    timings_external_alignment = True

    def __init__(
        self,
        routes: Mapping[str, Sequence[Any] | Any],
        *,
        default_language: str = "en",
        pre_first_pcm_timeout: float = 2.0,
        stop_timeout: float = 2.0,
    ) -> None:
        if not isinstance(routes, Mapping) or not routes:
            raise ValueError("routes must be a non-empty mapping")
        try:
            pre_first_pcm_timeout = float(pre_first_pcm_timeout)
            stop_timeout = float(stop_timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("routing timeouts must be finite and positive") from exc
        if (
            not math.isfinite(pre_first_pcm_timeout)
            or not math.isfinite(stop_timeout)
            or pre_first_pcm_timeout <= 0
            or stop_timeout <= 0
        ):
            raise ValueError("routing timeouts must be finite and positive")
        self._default_language = normalize_language(default_language)
        if self._default_language == "*":
            raise ValueError("default_language must be a concrete routing language")
        self._pre_first_pcm_timeout = pre_first_pcm_timeout
        self._stop_timeout = stop_timeout
        self._configured_entries = {}
        self._children = []
        self._child_ids = set()
        for raw_language, raw_entries in routes.items():
            language = normalize_language(raw_language)
            entries = self._coerce_entries(raw_entries)
            if not entries:
                raise ValueError(f"route {raw_language!r} has no providers")
            if language in self._configured_entries:
                raise ValueError(f"duplicate route for canonical language {language!r}")
            self._configured_entries[language] = tuple(entries)
            for entry in entries:
                if id(entry.engine) not in self._child_ids:
                    self._child_ids.add(id(entry.engine))
                    self._children.append(entry.engine)
        if not self._children:
            raise ValueError("routes must contain at least one provider")
        self.routes = MappingProxyType({
            language: tuple(entries)
            for language, entries in self._configured_entries.items()
        })
        self._format_info = None

    def post_init(self):
        self.engine_name = "language-routing"
        # Fallback replays text to another child; generators cannot be replayed.
        self.can_consume_generators = False
        self.direct_voice_control = any(
            bool(getattr(child, "direct_voice_control", False))
            for child in self._children
        )
        self.instruction_control = any(
            bool(getattr(child, "instruction_control", False))
            for child in self._children
        )
        self._validate_children()
        self.provides_word_timings = all(
            bool(getattr(child, "provides_word_timings", False))
            and hasattr(child, "on_alignment")
            for child in self._children
        )
        self.streaming_aligner = None
        self.timings_external_alignment = not self.provides_word_timings
        self.on_alignment = None
        self.last_alignment = None
        self._alignment_bridges = set()
        self._quarantined: dict[int, _Attempt] = {}
        self._synth_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._active_attempt = None
        self._generation = 0
        self._closed = False
        self._language = self._default_language
        self._logical_voice = None
        self._instructions = ""
        self._stream_context = None

    @staticmethod
    def _coerce_entries(raw_entries):
        if isinstance(raw_entries, LanguageRoute):
            return [raw_entries]
        if isinstance(raw_entries, Sequence) and not isinstance(
            raw_entries, (str, bytes, bytearray)
        ):
            values = list(raw_entries)
        else:
            values = [raw_entries]
        return [
            value if isinstance(value, LanguageRoute) else LanguageRoute(value)
            for value in values
        ]

    @staticmethod
    def _provider_name(provider):
        name = getattr(provider, "engine_name", None)
        if isinstance(name, str) and name and name != "unknown":
            return name
        return provider.__class__.__name__

    def _validate_children(self):
        first_info = None
        for child in self._children:
            if not callable(getattr(child, "synthesize", None)):
                raise TypeError(f"{self._provider_name(child)} has no synthesize()")
            if not hasattr(child, "queue"):
                raise TypeError(f"{self._provider_name(child)} has no output queue")
            info = tuple(child.get_stream_info())
            if len(info) != 3 or info != SUPPORTED_STREAM_INFO:
                raise ValueError(
                    f"{self._provider_name(child)} exposes {info!r}; "
                    f"language routing requires {SUPPORTED_STREAM_INFO!r}"
                )
            if first_info is not None and info != first_info:
                raise ValueError("all routed providers must expose the same stream format")
            first_info = info
        self._format_info = first_info or SUPPORTED_STREAM_INFO

    def _route_entries(self, language):
        entries = self._configured_entries.get(language, ())
        if not entries and language != "*":
            entries = self._configured_entries.get("*", ())
        result = []
        seen = set()
        for entry in entries:
            if id(entry.engine) not in seen:
                seen.add(id(entry.engine))
                result.append(entry)
        return tuple(result)

    def routing_metadata(self):
        return {
            "default_language": self._default_language,
            "routes": {
                language: [self._provider_name(entry.engine) for entry in entries]
                for language, entries in self._configured_entries.items()
            },
            "fallback": {
                "phase": "before_first_pcm",
                "timeout_seconds": self._pre_first_pcm_timeout,
            },
            "timings_external_alignment": self.timings_external_alignment,
            "provides_word_timings": self.provides_word_timings,
        }

    def get_stream_info(self):
        return self._format_info

    def get_language(self):
        with self._state_lock:
            return self._language

    def set_language(self, language):
        with self._state_lock:
            self._language = normalize_language(language, allow_auto=True)

    def set_voice(self, voice):
        with self._state_lock:
            self._logical_voice = (
                None
                if voice is None or (isinstance(voice, str) and voice == "default")
                else voice
            )

    def set_inline_voice(self, voice):
        self.set_voice(voice)

    def get_instructions(self):
        with self._state_lock:
            return self._instructions

    def set_instructions(self, instructions):
        if not isinstance(instructions, str):
            raise TypeError("instructions must be a string")
        with self._state_lock:
            self._instructions = instructions

    def set_voice_parameters(self, **voice_parameters):
        unsupported = set(voice_parameters) - {"language", "instructions"}
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(
                f"LanguageRoutingEngine does not own provider parameters: {names}"
            )
        if "language" in voice_parameters:
            self.set_language(voice_parameters["language"])
        if "instructions" in voice_parameters:
            self.set_instructions(voice_parameters["instructions"])

    def _logical_state(self):
        with self._state_lock:
            return self._language, self._logical_voice, self._instructions

    @staticmethod
    def _normalize_non_speech_ranges(ranges):
        if not isinstance(ranges, (list, tuple)):
            return None
        normalized = []
        for span in ranges:
            if not isinstance(span, Mapping):
                return None
            start, end = span.get("start"), span.get("end")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end < start
            ):
                return None
            normalized.append({"start": start, "end": end})
        return normalized

    def alignment_non_speech_ranges(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        language, _, _ = self._logical_state()
        concrete = self._default_language if language == "auto" else language
        entries = self._route_entries(concrete)
        if not entries:
            return []
        candidates = []
        for entry in entries:
            method = getattr(entry.engine, "alignment_non_speech_ranges", None)
            if not callable(method):
                return []
            try:
                ranges = method(text)
            except Exception:
                return []
            normalized = self._normalize_non_speech_ranges(ranges)
            if normalized is None:
                return []
            candidates.append(normalized)
        first = candidates[0]
        if any(candidate != first for candidate in candidates[1:]):
            return []
        return [dict(span) for span in first]

    def _mapped_voice(self, entry, logical_voice):
        if logical_voice is None:
            return entry.voice
        if entry.voices is not None:
            try:
                return entry.voices[logical_voice]
            except (KeyError, TypeError) as exc:
                raise _NoPcmFailure(
                    f"{self._provider_name(entry.engine)} has no voice mapping for "
                    f"{logical_voice!r}"
                ) from exc
        return logical_voice

    @staticmethod
    def _language_name(language):
        return _CANONICAL_LANGUAGE_NAMES.get(language)

    def _apply_provider_language(self, provider, language):
        setter = getattr(provider, "set_language", None)
        if not callable(setter):
            raise _NoPcmFailure(
                f"{self._provider_name(provider)} cannot select language {language!r}"
            )
        try:
            setter(language)
            return
        except Exception as first_error:
            native_name = self._language_name(language)
            if native_name is None or native_name == language:
                raise _NoPcmFailure(
                    f"{self._provider_name(provider)} rejected language {language!r}"
                ) from first_error
            try:
                setter(native_name)
            except Exception as second_error:
                raise _NoPcmFailure(
                    f"{self._provider_name(provider)} rejected language {language!r}"
                ) from second_error

    def _apply_provider_state(self, entry, language, logical_voice, instructions):
        provider = entry.engine
        mapped_voice = self._mapped_voice(entry, logical_voice)
        if mapped_voice is not None:
            setter = getattr(provider, "set_inline_voice", None)
            if not callable(setter):
                setter = getattr(provider, "set_voice", None)
            if not callable(setter):
                raise _NoPcmFailure(
                    f"{self._provider_name(provider)} cannot select a voice"
                )
            try:
                setter(mapped_voice)
            except Exception as exc:
                raise _NoPcmFailure(
                    f"{self._provider_name(provider)} rejected voice selection"
                ) from exc
        self._apply_provider_language(provider, language)
        if getattr(provider, "direct_voice_control", False):
            setter = getattr(provider, "set_instructions", None)
            if callable(setter):
                try:
                    setter(instructions)
                except Exception as exc:
                    raise _NoPcmFailure(
                        f"{self._provider_name(provider)} rejected instructions"
                    ) from exc

    @staticmethod
    def _translate_alignment_event(event, offset):
        if not isinstance(event, Mapping):
            return event
        translated = dict(event)
        if "engine_audio_offset" in translated:
            try:
                translated["engine_audio_offset"] = (
                    float(translated["engine_audio_offset"]) + float(offset)
                )
            except (TypeError, ValueError):
                pass
        return translated

    def _alignment_bridge(self, provider, attempt):
        if not hasattr(provider, "on_alignment"):
            raise _NoPcmFailure(
                f"{self._provider_name(provider)} has no native alignment callback"
            )
        original = getattr(provider, "on_alignment", None)
        bridge = _AlignmentBridge(
            self,
            provider,
            original,
            getattr(self, "on_alignment", None),
            attempt,
            attempt.alignment_offset,
        )
        self._alignment_bridges.add(bridge)
        attempt.alignment_bridge = bridge
        provider.on_alignment = bridge
        return bridge

    def _release_alignment_bridge(self, bridge):
        with self._state_lock:
            self._alignment_bridges.discard(bridge)

    def _fence_alignment_bridges(self, *, restore, context=_ALL_CONTEXTS):
        del restore
        for bridge in tuple(self._alignment_bridges):
            if (
                context is not _ALL_CONTEXTS
                and bridge.attempt is not None
                and bridge.attempt.context is not context
            ):
                continue
            bridge.fence()
            provider = bridge.provider
            if getattr(provider, "on_alignment", None) is bridge:
                provider.on_alignment = bridge.original

    def _is_active_attempt(self, attempt):
        with self._state_lock:
            return self._active_attempt is attempt

    def _child_context(self, provider, context=_CURRENT_CONTEXT):
        if context is _CURRENT_CONTEXT:
            context = self._stream_context
        if context is None:
            return None
        if not isinstance(context, _RoutingStreamContext):
            return context
        identity = id(provider)
        if identity not in context.contexts:
            creator = getattr(provider, "create_stream_context", None)
            context.contexts[identity] = creator() if callable(creator) else None
        return context.contexts[identity]

    def create_stream_context(self):
        context = _RoutingStreamContext()
        for child in self._children:
            creator = getattr(child, "create_stream_context", None)
            context.contexts[id(child)] = creator() if callable(creator) else None
        return context

    def set_stream_context(self, context):
        self._stream_context = context
        active = self._active_attempt
        if active is not None and context is active.context:
            setter = getattr(active.provider, "set_stream_context", None)
            if callable(setter):
                setter(self._child_context(active.provider, active.context))

    def close_stream_context(self, context):
        active = self._active_attempt
        if active is not None and active.context is context:
            active.cancelled.set()
            self._request_provider_stop(active)
        self._fence_alignment_bridges(restore=True, context=context)
        if isinstance(context, _RoutingStreamContext):
            context.closed = True
            errors = []
            for child in self._children:
                close = getattr(child, "close_stream_context", None)
                child_context = context.contexts.get(id(child))
                if callable(close) and child_context is not None:
                    try:
                        close(child_context)
                    except BaseException as exc:
                        errors.append(exc)
            if self._stream_context is context:
                self._stream_context = None
            if errors:
                raise errors[0]
            return
        close = getattr(active.provider, "close_stream_context", None) if active else None
        if callable(close):
            close(context)
        if self._stream_context is context:
            self._stream_context = None

    def _invoke_provider(self, attempt, text, sentence_count):
        try:
            attempt.result = attempt.provider.synthesize(text, sentence_count)
        except BaseException as exc:
            attempt.error = exc

    def _request_provider_stop(self, attempt):
        if attempt.stop_requested:
            return
        attempt.stop_requested = True
        stopper = getattr(attempt.provider, "stop", None)
        if callable(stopper):
            try:
                stopper()
            except BaseException:
                pass

    def _restore_attempt_sinks(self, attempt):
        provider = attempt.provider
        if attempt.queue_sink is not None and getattr(provider, "queue", None) is attempt.queue_sink:
            provider.queue = attempt.original_queue
        if attempt.has_timings and getattr(provider, "timings", None) is attempt.timings_sink:
            provider.timings = attempt.original_timings

    def _quarantine_attempt(self, attempt):
        provider = attempt.provider
        with self._state_lock:
            if self._quarantined.get(id(provider)) is attempt:
                return
            self._quarantined[id(provider)] = attempt
        threading.Thread(
            target=self._wait_for_quarantined_attempt,
            args=(attempt,),
            name=f"{self.engine_name}:{self._provider_name(provider)}-quarantine",
            daemon=True,
        ).start()

    def _wait_for_quarantined_attempt(self, attempt):
        thread = attempt.thread
        if thread is not None:
            thread.join()
        self._restore_attempt_sinks(attempt)
        with self._state_lock:
            if self._quarantined.get(id(attempt.provider)) is attempt:
                self._quarantined.pop(id(attempt.provider), None)

    def _run_provider(
        self, entry, language, logical_voice, instructions, text, sentence_count
    ):
        attempt = _Attempt(entry.engine, self._generation)
        provider = entry.engine
        with self._state_lock:
            if id(provider) in self._quarantined:
                raise _NoPcmFailure(
                    f"{self._provider_name(provider)} is quarantined after a stalled worker"
                )
            attempt.context = self._stream_context
            self._active_attempt = attempt
        if (
            isinstance(attempt.context, _RoutingStreamContext)
            and attempt.context.closed
        ):
            with self._state_lock:
                if self._active_attempt is attempt:
                    self._active_attempt = None
            raise _NoPcmFailure("stream context is closed")
        attempt.original_queue = provider.queue
        sentinel = object()
        attempt.original_timings = getattr(provider, "timings", sentinel)
        attempt.has_timings = attempt.original_timings is not sentinel
        provider = attempt.provider
        bridge = None
        try:
            if attempt.cancelled.is_set():
                raise _AttemptCancelled
            self._apply_provider_state(entry, language, logical_voice, instructions)
            if attempt.cancelled.is_set():
                raise _AttemptCancelled
            attempt.router_audio_start = float(getattr(self, "audio_duration", 0) or 0)
            attempt.child_audio_start = float(getattr(provider, "audio_duration", 0) or 0)
            attempt.queue_sink = _FencedSink(self, attempt, self.queue, commit_pcm=True)
            provider.queue = attempt.queue_sink
            if attempt.has_timings:
                attempt.timings_sink = _FencedSink(
                    self,
                    attempt,
                    self.timings,
                    commit_pcm=False,
                    drop_if_native_callback=True,
                )
                provider.timings = attempt.timings_sink
            if self.provides_word_timings:
                attempt.alignment_offset = attempt.router_audio_start - attempt.child_audio_start
                bridge = self._alignment_bridge(provider, attempt)
            setter = getattr(provider, "set_stream_context", None)
            if callable(setter) and attempt.context is not None:
                setter(self._child_context(provider, attempt.context))
            thread = threading.Thread(
                target=self._invoke_provider,
                args=(attempt, text, sentence_count),
                name=f"{self.engine_name}:{self._provider_name(provider)}",
                daemon=True,
            )
            attempt.thread = thread
            thread.start()
            deadline = time.monotonic() + self._pre_first_pcm_timeout
            while thread.is_alive() and not attempt.first_pcm.is_set():
                if attempt.cancelled.is_set():
                    self._request_provider_stop(attempt)
                    break
                if time.monotonic() >= deadline:
                    attempt.timed_out = True
                    attempt.cancelled.set()
                    self._request_provider_stop(attempt)
                    thread.join(self._stop_timeout)
                    if thread.is_alive():
                        raise TimeoutError(
                            f"{self._provider_name(provider)} did not stop after timeout"
                        )
                    break
                attempt.first_pcm.wait(0.02)
            if attempt.cancelled.is_set():
                if thread.is_alive():
                    self._request_provider_stop(attempt)
                    thread.join(self._stop_timeout)
                if attempt.timed_out:
                    if attempt.first_pcm.is_set():
                        raise RuntimeError(
                            f"{self._provider_name(provider)} timed out while delivering PCM"
                        )
                    raise _NoPcmFailure(
                        f"{self._provider_name(provider)} timed out before first PCM"
                    )
                raise _AttemptCancelled
            while thread.is_alive() and not attempt.cancelled.is_set():
                thread.join(0.02)
            if attempt.cancelled.is_set():
                raise _AttemptCancelled
            if attempt.error is not None:
                if attempt.first_pcm.is_set():
                    raise attempt.error
                raise _NoPcmFailure(
                    f"{self._provider_name(provider)} failed before first PCM"
                ) from attempt.error
            if not attempt.first_pcm.is_set():
                raise _NoPcmFailure(
                    f"{self._provider_name(provider)} produced no PCM"
                )
            if attempt.result is False:
                raise RuntimeError(
                    f"{self._provider_name(provider)} failed after producing PCM"
                )
            provider_end = float(getattr(provider, "audio_duration", 0) or 0)
            child_start = attempt.child_audio_start
            provider_delta = provider_end - child_start
            if provider_delta < 0:
                provider_delta = provider_end
            pcm_delta = attempt.pcm_bytes / (2 * SUPPORTED_STREAM_INFO[2])
            self.audio_duration = attempt.router_audio_start + (
                pcm_delta if pcm_delta > 0 else max(0.0, provider_delta)
            )
            self.last_alignment = getattr(provider, "last_alignment", None)
            attempt.alignment_success = True
            return True
        finally:
            if bridge is not None:
                if not attempt.alignment_success:
                    bridge.fence()
                if getattr(provider, "on_alignment", None) is bridge:
                    provider.on_alignment = bridge.original
            attempt.cancelled.set()
            thread = attempt.thread
            if thread is not None and thread.is_alive():
                self._request_provider_stop(attempt)
                thread.join(self._stop_timeout)
            lingering = thread is not None and thread.is_alive()
            with self._state_lock:
                if self._active_attempt is attempt:
                    self._active_attempt = None
            if lingering:
                self._quarantine_attempt(attempt)
            else:
                self._restore_attempt_sinks(attempt)

    def synthesize(self, text, sentence_count=0):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        with self._synth_lock:
            if self._closed:
                raise RuntimeError("LanguageRoutingEngine is shut down")
            self.stop_synthesis_event.clear()
            self._generation += 1
            language, logical_voice, instructions = self._logical_state()
            concrete = self._default_language if language == "auto" else language
            entries = self._route_entries(concrete)
            if not entries:
                raise ValueError(f"no providers configured for language {concrete!r}")
            last_failure = None
            for entry in entries:
                if self.stop_synthesis_event.is_set():
                    return False
                try:
                    return self._run_provider(
                        entry, concrete, logical_voice, instructions, text, sentence_count
                    )
                except _AttemptCancelled:
                    return False
                except _NoPcmFailure as exc:
                    last_failure = exc
            if last_failure is not None:
                raise RuntimeError(
                    f"all providers failed before first PCM for {concrete!r}"
                ) from last_failure
            raise RuntimeError(f"all providers failed for {concrete!r}")

    def pause(self, timeout=2.0):
        active = self._active_attempt
        if active is None:
            return True
        pauser = getattr(active.provider, "pause", None)
        return bool(pauser(timeout=timeout)) if callable(pauser) else False

    def resume(self):
        active = self._active_attempt
        if active is None:
            return True
        resumer = getattr(active.provider, "resume", None)
        if not callable(resumer):
            return False
        result = resumer()
        return True if result is None else bool(result)

    def stop(self):
        self.stop_synthesis_event.set()
        active = self._active_attempt
        context = active.context if active is not None else self._stream_context
        if active is not None:
            active.cancelled.set()
            self._request_provider_stop(active)
        self._fence_alignment_bridges(restore=False, context=context)

    def shutdown(self):
        self._closed = True
        self.stop()
        self._fence_alignment_bridges(restore=True)
        self._stream_context = None

    def reset_audio_duration(self):
        self.audio_duration = 0
        for child in getattr(self, "_children", ()):
            reset = getattr(child, "reset_audio_duration", None)
            if callable(reset):
                reset()

    def get_voices(self):
        logical_names = []
        seen = set()
        for entries in self._configured_entries.values():
            for entry in entries:
                if entry.voices:
                    for name in entry.voices:
                        try:
                            if name not in seen:
                                seen.add(name)
                                logical_names.append(name)
                        except TypeError:
                            pass
        if logical_names:
            return logical_names
        for child in self._children:
            getter = getattr(child, "get_voices", None)
            if callable(getter):
                try:
                    return list(getter())
                except NotImplementedError:
                    pass
        return []


__all__ = [
    "LanguageRoute",
    "LanguageRoutingEngine",
    "SUPPORTED_STREAM_INFO",
    "normalize_language",
]
