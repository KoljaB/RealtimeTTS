"""Streaming adapter for an existing RealtimeTTS HTTP/WebSocket service.

A logical stream keeps one provider connection, including native context and
asynchronous alignment. The outer server owns sentence splitting and pauses.
"""
from __future__ import annotations

import json
import logging
import math
import re
from contextlib import suppress
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import requests
import websocket

from .._audio_formats import INT16
from ..language_router import normalize_qwen_language
from .base_engine import BaseEngine, TimingInfo

LOGGER = logging.getLogger(__name__)

_SIGNED64_MIN = -(2**63)
_SIGNED64_MAX = 2**63 - 1
_MAX_NEW_TOKENS = 2**31 - 1


def _utf16_offset(text, index):
    return len(text[:index].encode("utf-16-le")) // 2


def _validate_sampling_parameter(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{name} is out of domain")
    return value


class _Alignment:
    def __init__(self, callback, offset):
        self.callback, self.offset = callback, offset
        self.done = threading.Event()
        self.closed = False

    def publish(self, event):
        if not self.closed and self.callback is not None:
            try:
                self.callback({**event, "engine_audio_offset": self.offset})
            except Exception:
                LOGGER.exception("Provider alignment callback failed")
        if event.get("type") == "alignment_done":
            self.done.set()

    def finish(self, *, cancelled=False):
        if not self.done.is_set():
            self.publish({"type": "alignment_done", "complete": False,
                          "finish_reason": "cancelled" if cancelled else "unavailable"})
        self.closed = True

    def close(self):
        self.closed = True


class _Fragment:
    def __init__(self, engine, segment):
        self.segment = segment
        self.queue, self.timings = engine.queue, engine.timings
        self.offset = engine.audio_duration
        self.alignment = _Alignment(engine.on_alignment, self.offset)
        if not engine.provides_word_timings:
            self.alignment.done.set()
        self.done = threading.Event()
        self.bytes = 0
        self.error = None
        self.request_id = None


class _Connection:
    def __init__(self, engine, config):
        self.engine, self.config = engine, config
        self.closed = threading.Event()
        self.paused = threading.Event()
        self.finished = threading.Event()
        self.lock = threading.RLock()
        self.fragments, self.requests = {}, {}
        self.active = None
        self.serial = 0
        headers = {"Authorization": "Bearer " + engine._api_key} if engine._api_key else {}
        self.socket = websocket.create_connection(
            engine.ws_url, header=headers, timeout=min(engine.timeout, 5.0),
            enable_multithread=True,
        )
        self.socket.settimeout(engine.timeout)
        try:
            self.send({"type": "config", **config})
        except BaseException:
            self.socket.close()
            raise
        self.reader = threading.Thread(target=self._read, name="tts-provider-stream", daemon=True)
        self.reader.start()

    def send(self, event):
        with self.lock:
            if self.closed.is_set():
                raise RuntimeError("provider connection is closed")
            self.socket.send(json.dumps(event, ensure_ascii=False))

    def start(self, text):
        with self.lock:
            if self.active is not None:
                raise RuntimeError("provider stream already has an active fragment")
            self.serial += 1
            fragment = _Fragment(self.engine, self.serial)
            self.fragments[fragment.segment] = fragment
            self.active = fragment
            self.engine.last_alignment = fragment.alignment
            self.send({"type": "segment_start", "segment_id": fragment.segment})
            event = {"type": "text", "text": text, "voice": self.engine.voice}
            if self.engine.features.get("per_text_language"):
                event["language"] = self.engine.language
            if self.engine.direct_voice_control:
                event["instructions"] = self.engine.instructions or ""
            self.send(event)
            self.send({"type": "flush", "segment_id": fragment.segment})
            return fragment

    def _retire(self, fragment):
        if fragment.done.is_set() and fragment.alignment.done.is_set():
            self.fragments.pop(fragment.segment, None)
            self.requests.pop(fragment.request_id, None)

    def _read(self):
        error = None
        try:
            while not self.closed.is_set():
                message = self.socket.recv()
                if not message:
                    raise RuntimeError("provider closed before completing the stream")
                if isinstance(message, bytes):
                    fragment = self.active
                    if fragment is None or fragment.done.is_set():
                        raise RuntimeError("provider sent PCM outside an active segment")
                    if len(message) % 2:
                        raise RuntimeError("provider returned an incomplete PCM frame")
                    if not self.engine.stop_synthesis_event.is_set():
                        fragment.queue.put(message)
                        fragment.bytes += len(message)
                        self.engine.audio_duration += len(message) / 48000
                    continue
                event = json.loads(message)
                kind = event.get("type")
                if kind == "error":
                    raise RuntimeError(str(event.get("error", {}).get("message", "provider synthesis failed")))
                if kind in {"done", "cancelled"}:
                    self.finished.set()
                    break
                if kind == "paused":
                    self.paused.set()
                elif kind == "resumed":
                    self.paused.clear()
                elif kind == "fragment_start":
                    fragment = self.active
                    if fragment is None:
                        raise RuntimeError("provider alignment has no active fragment")
                    fragment.request_id = event["request_id"]
                    self.requests[fragment.request_id] = fragment
                elif kind == "segment_done":
                    fragment = self.fragments.get(event.get("segment_id"))
                    if fragment is not None:
                        if event.get("skipped"):
                            fragment.error = RuntimeError("provider skipped the fragment")
                        self.active = None
                        fragment.done.set()
                        self._retire(fragment)
                elif kind in {"alignment", "alignment_done", "alignment_error", "alignment_complete"}:
                    fragment = self.requests.get(event.get("request_id"))
                    if fragment is not None:
                        fragment.alignment.publish(event)
                        if fragment.alignment.callback is None and not fragment.alignment.closed:
                            for word in event.get("words", ()):
                                fragment.timings.put(TimingInfo(
                                    fragment.offset + word["start_time"],
                                    fragment.offset + word["end_time"], word["word"],
                                ))
                        self._retire(fragment)
        except Exception as exc:
            error = exc
        finally:
            self.closed.set()
            for fragment in list(self.fragments.values()):
                fragment.error = error or RuntimeError("provider stream ended")
                fragment.alignment.finish(cancelled=True)
                fragment.done.set()
            self.finished.set()

    def close(self, *, cancel=False):
        if not self.closed.is_set():
            try:
                self.send({"type": "cancel" if cancel else "end"})
            except Exception:
                pass
        self.closed.set()
        # shutdown wakes a blocking recv without waiting for the network timeout.
        with suppress(OSError, websocket.WebSocketException):
            self.socket.shutdown()
        with suppress(OSError, websocket.WebSocketException):
            self.socket.close(timeout=0)
        if threading.current_thread() is not self.reader:
            self.reader.join(timeout=1)


class _StreamContext:
    def __init__(self):
        self.connection = None
        self.closed = False


class RemoteTTSEngine(BaseEngine):
    """Use a RealtimeTTS server as a streaming, controllable route provider.

    Both services must advertise per_request_silence and segmented narration.
    Native provider alignment stays asynchronous; PCM is never buffered for it.
    Caller-owned remote services are never shut down by this client.
    """

    def __init__(self, base_url, voice, *, api_key=None, language="auto",
                 instructions=None, timeout=120.0, seed=None, max_new_tokens=None):
        parsed = urlsplit(base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an HTTP(S) service URL")
        if parsed.username or parsed.password:
            raise ValueError("Use api_key rather than credentials in the service URL")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        seed = _validate_sampling_parameter("seed", seed, _SIGNED64_MIN, _SIGNED64_MAX)
        max_new_tokens = _validate_sampling_parameter(
            "max_new_tokens", max_new_tokens, 1, _MAX_NEW_TOKENS
        )
        self.base_url, self.timeout = base_url.rstrip("/"), float(timeout)
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.ws_url = urlunsplit(("wss" if parsed.scheme == "https" else "ws",
                                 parsed.netloc, parsed.path + "/v1/audio/speech-stream", "", ""))
        self._api_key = api_key
        headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        response = requests.get(self.base_url + "/v1/capabilities", headers=headers, timeout=self.timeout)
        response.raise_for_status()
        self.capabilities = response.json()
        self.features = self.capabilities.get("features", {})
        self._vocal_tag_pattern = None
        if self.features.get("alignment_non_speech_ranges") == "utf16_code_units":
            vocal_tags = self.features.get("vocal_tags")
            if isinstance(vocal_tags, (list, tuple)):
                tags = tuple(dict.fromkeys(
                    tag for tag in vocal_tags if isinstance(tag, str) and tag
                ))
                if tags:
                    self._vocal_tag_pattern = re.compile(
                        r"\((" + "|".join(
                            re.escape(tag) for tag in sorted(tags, key=len, reverse=True)
                        ) + r")\)",
                        re.IGNORECASE,
                    )
        controls = self.capabilities.get("stream_controls", {})
        if not self.features.get("per_request_silence") or not controls.get("segmented_narration"):
            raise ValueError("Upgrade the provider server: routing requires per_request_silence and segmented_narration")
        audio = self.capabilities.get("audio", {})
        if (audio.get("sample_rate_hz") != 24000 or audio.get("channels") != 1
                or audio.get("encoding") != "s16le"):
            raise ValueError("Routing requires provider PCM s16le, mono, 24000 Hz")
        self.direct_voice_control = bool(self.features.get("direct_voice_control"))
        self.instruction_control = self.direct_voice_control
        self._default_instructions = instructions
        self._delivery_instructions = None
        self.instructions = instructions
        self.set_voice(voice)
        self.set_language(language)
        self._context = None
        self._active = None
        self._last_connection = None
        self._generation_lock = threading.Lock()
        self._control = threading.RLock()
        self._pause_requested = threading.Event()
        self.on_alignment = None
        self.last_alignment = None
        self.last_error = None

    def post_init(self):
        self.engine_name = "remote-tts"
        self.provides_word_timings = bool(self.features.get("streaming_alignment"))

    def get_stream_info(self):
        return INT16, 1, 24000

    def get_language(self):
        return self.language

    def alignment_non_speech_ranges(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        pattern = self._vocal_tag_pattern
        if pattern is None:
            return []
        return [
            {
                "start": _utf16_offset(text, match.start()),
                "end": _utf16_offset(text, match.end()),
            }
            for match in pattern.finditer(text)
        ]

    def set_language(self, language):
        self.language = normalize_qwen_language(language, allow_auto=True)

    def set_voice(self, voice):
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError("voice must name a provider voice")
        self.voice = voice.strip()

    def get_voices(self):
        headers = {"Authorization": "Bearer " + self._api_key} if self._api_key else {}
        response = requests.get(self.base_url + "/v1/audio/voices", headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return [voice["name"] for voice in response.json()["voices"]]

    def get_instructions(self):
        """Return the delivery override accepted by set_instructions."""
        return self._delivery_instructions

    def set_instructions(self, instructions):
        if instructions is not None and not isinstance(instructions, str):
            raise TypeError("instructions must be a string or None")
        if not self.direct_voice_control and instructions:
            raise ValueError("provider does not support delivery instructions")
        direction = instructions.strip() if instructions else None
        self._delivery_instructions = direction
        design = self.features.get("voice_design_voice") == self.voice
        self.instructions = (
            self._default_instructions + " Delivery: " + direction
            if design and self._default_instructions and direction
            else direction or self._default_instructions
        )

    def set_voice_parameters(self, **parameters):
        unknown = parameters.keys() - {"language", "instructions"}
        if unknown:
            raise ValueError("Configure sampling on each provider, not the routing client")
        if "language" in parameters:
            self.set_language(parameters["language"])
        if "instructions" in parameters:
            self.set_instructions(parameters["instructions"])

    def create_stream_context(self):
        return _StreamContext()

    def set_stream_context(self, context):
        self._context = context

    def close_stream_context(self, context):
        context.closed = True
        if context.connection is not None:
            context.connection.close()
            context.connection = None

    def _connection(self, context):
        if context.closed:
            raise RuntimeError("provider stream context is closed")
        connection = context.connection
        if connection is not None and (
            connection.closed.is_set()
            or (not self.features.get("per_text_language")
                and connection.config["language"] != self.language)
        ):
            connection.close()
            connection = context.connection = None
        if connection is None:
            config = {
                "voice": self.voice, "language": self.language, "response_format": "pcm",
                "split_sentences": False, "silence": {"comma": 0, "sentence": 0, "transition": 0},
            }
            if self.seed is not None:
                config["seed"] = self.seed
            if self.max_new_tokens is not None:
                config["max_new_tokens"] = self.max_new_tokens
            if self.direct_voice_control:
                config["instructions"] = self.instructions
            connection = context.connection = _Connection(self, config)
        return connection

    def synthesize(self, text, sentence_count=0):
        del sentence_count
        with self._generation_lock:
            self.stop_synthesis_event.clear()
            self.last_error = None
            owns_context = self._context is None
            context = self._context or self.create_stream_context()
            fragment = None
            try:
                with self._control:
                    connection = self._connection(context)
                    self._active = self._last_connection = connection
                    state_cancel = getattr(getattr(self.queue, "state", None), "cancelled", None)
                    if self.stop_synthesis_event.is_set() or (state_cancel is not None and state_cancel.is_set()):
                        connection.close(cancel=True)
                        return False
                    if self._pause_requested.is_set():
                        connection.send({"type": "pause"})
                    fragment = connection.start(text)
                if not fragment.done.wait(self.timeout):
                    raise TimeoutError("provider synthesis timed out")
                if fragment.error:
                    raise fragment.error
                return fragment.bytes > 0 and not self.stop_synthesis_event.is_set()
            except Exception as exc:
                self.last_error = exc
                if context.connection is not None:
                    context.connection.close(cancel=True)
                return False
            finally:
                with self._control:
                    self._active = None
                if owns_context:
                    # Standalone calls retain their metadata until completion.
                    if fragment is not None and not self.stop_synthesis_event.is_set():
                        fragment.alignment.done.wait(min(self.timeout, 5.0))
                    self.close_stream_context(context)

    def stop(self):
        self.stop_synthesis_event.set()
        self._pause_requested.clear()
        connection = self._active
        if connection is not None:
            connection.close(cancel=True)

    def pause(self, timeout=2.0):
        self._pause_requested.set()
        connection = self._active
        if connection is None or connection.active is None:
            return True
        connection.send({"type": "pause"})
        deadline = time.monotonic() + timeout
        while not connection.paused.is_set():
            if connection.closed.is_set() or connection.active is None:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            connection.paused.wait(min(.01, remaining))
        return True

    def resume(self):
        self._pause_requested.clear()
        connection = self._active or self._last_connection
        if connection is not None and not connection.closed.is_set():
            connection.send({"type": "resume"})

    def shutdown(self):
        self.stop()
