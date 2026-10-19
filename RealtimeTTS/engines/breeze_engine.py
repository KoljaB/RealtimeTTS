"""Breeze TTS 2 streaming engine for the local Breeze-TTS-2.cpp worker.

The worker owns the model/GPU. The Python engine owns voice registration, PCM
delivery and acknowledged native pause/cancel control.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests
import websocket

from .base_engine import BaseEngine, TimingInfo

LOGGER = logging.getLogger(__name__)
SAMPLE_RATE = 24000
VOCAL_TAGS = (
    "laugh", "chuckle", "giggle", "cry", "sob", "whimper", "groan", "moan",
    "sigh", "gasp", "inhale", "exhale", "breathe heavily", "clear throat",
    "clears throat", "cough", "sniffle", "lip smack", "click tongue", "yawn",
    "sneeze", "hiccup", "burp", "gulp", "gag", "grunt", "scoff", "snort",
)
_SOUND_NAMES = "|".join(re.escape(tag).replace(r"\ ", r"[ _]") for tag in VOCAL_TAGS)
_DIRECTION = re.compile(
    r"\[(?P<sound>" + _SOUND_NAMES + r")\]"
    r"|(?:^|(?<=[\r\n]))[ \t]*\[(?P<tone>[^\]\r\n]{1,500})\][ \t\r\n]*",
    re.IGNORECASE,
)
_INLINE_VOCAL_TAG = re.compile(
    r"\((" + "|".join(re.escape(tag) for tag in VOCAL_TAGS) + r")\)"
    r"|\[(" + _SOUND_NAMES + r")\]",
    re.IGNORECASE,
)
_VOCAL_DIRECTIONS = {
    "chuckle": "Chuckle briefly.",
    "laugh": "Brief laughter.",
    "clear throat": "Clear your throat once, briefly.",
    "sigh": "Exhale a gentle sigh.",
}


class BreezeEngineError(RuntimeError):
    pass


@dataclass
class BreezeVoice:
    name: str
    ref_audio: str | Path | None = None
    ref_text: str | None = None
    instruct: str | None = None
    language: str = "auto"
    voice_id: str | None = None
    voice_design: bool = False

    @classmethod
    def builtin(cls, language: str = "en") -> "BreezeVoice":
        """Select a packaged English or Chinese reference with its transcript."""
        from ..voices import bundled_voice_data
        return cls(**bundled_voice_data("breeze", language))


@dataclass
class _BreezeStreamContext:
    socket: Any = None
    started: bool = False
    signature: tuple | None = None
    closed: bool = False
    commands: list[dict] = field(default_factory=list)
    blocks: list[dict] = field(default_factory=list)


def strip_inline_vocal_tags(text: str) -> str:
    """Remove only Breeze's canonical sound tags for forced alignment."""
    return _INLINE_VOCAL_TAG.sub(" ", text)


def _utf16_offset(text: str, index: int) -> int:
    return len(text[:index].encode("utf-16-le")) // 2


def alignment_non_speech_ranges(text: str) -> list[dict[str, int]]:
    """Return non-spoken tag spans using the synthesis parser's exact rules."""
    spans = {(match.start(), match.end())
             for pattern in (_INLINE_VOCAL_TAG, _DIRECTION)
             for match in pattern.finditer(text)}
    return [
        {"start": _utf16_offset(text, start), "end": _utf16_offset(text, end)}
        for start, end in sorted(spans)
    ]


def directed_blocks(text: str, instruction: str):
    """Expand [sound] shortcuts with a temporary direction, like the Python demo."""
    start = 0
    for tag in _DIRECTION.finditer(text):
        preceding = text[start:tag.start()].strip()
        if preceding:
            yield preceding, instruction
        if tag.group("sound"):
            sound = tag.group("sound").lower().replace("_", " ")
            yield f"({sound})", _VOCAL_DIRECTIONS.get(
                sound, f"Make a brief {sound} sound without speaking words.")
        else:
            instruction = tag.group("tone").strip()
        start = tag.end()
    remaining = text[start:].strip()
    if remaining:
        yield remaining, instruction


class BreezeEngine(BaseEngine):
    def __init__(
        self, voice: BreezeVoice | str | None = None, *,
        base_url: str = "http://127.0.0.1:8137",
        ws_url: str = "ws://127.0.0.1:8138",
        cfg_scale: float = 4.0, seed: int = 44, max_new_tokens: int = 500,
        temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0,
        repetition_penalty: float = 0.0, timeout: float = 120.0,
        require_native_controls: bool = True,
        reference_prefix_cache: bool = False,
        alignment_model: str | Path | None = None,
        alignment_language: str | None = None,
        alignment_threads: int = 2,
        streaming_aligner: Any = None,
        on_alignment: Callable[[dict[str, Any]], None] | None = None,
    ):
        if int(alignment_threads) <= 0:
            raise ValueError("alignment_threads must be positive")
        self.base_url, self.ws_url = base_url.rstrip("/"), ws_url
        self.cfg_scale, self.seed, self.max_new_tokens = cfg_scale, seed, max_new_tokens
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.repetition_penalty, self.timeout = repetition_penalty, timeout
        self.reference_prefix_cache = bool(reference_prefix_cache)
        self._initial_voice = voice
        self._voice = BreezeVoice("default")
        self._voice_id = ""
        self._prepared: dict[tuple, str] = {}
        self._control = threading.RLock()
        self._generation_lock = threading.Lock()
        self._socket = None
        self._stream_context = None
        self._socket_started = False
        self._cancel = threading.Event()
        self._pause_requested = threading.Event()
        self._paused = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self.last_error = None
        self.last_metrics: dict[str, Any] = {}
        self.last_diagnostics: dict[str, Any] = {}
        self._cancel_requested_at = None
        self.model_id = "BreezeBlue/Breeze-TTS-2"
        self.quant = "Q8_0"
        self.device = "unknown"
        self.model_type = "base"
        self.instruction_control = True
        self.direct_voice_control = True
        self.vocalizations = {tag.replace(" ", "_"): f"({tag})" for tag in VOCAL_TAGS}
        self.alignment_model = alignment_model
        self.alignment_language = alignment_language
        self.alignment_threads = int(alignment_threads)
        self.streaming_aligner = streaming_aligner
        self._owns_aligner = alignment_model is not None and streaming_aligner is None
        self.alignment_start_event = None
        self.alignment_cancel_event = None
        self.on_alignment = on_alignment
        # Breeze vocal inserts must remain in the output. Alignment is
        # observational and never allowed to request native tail cutoff.
        self.hallucination_guard = False
        self.last_alignment = None
        self.startup_buffer_ms = 250
        self._http = requests.Session()
        self._http.trust_env = False
        response = self._http.get(self.base_url + "/health", timeout=5)
        response.raise_for_status()
        health = response.json()
        self._worker_health = dict(health)
        if health.get("sample_rate") != SAMPLE_RATE:
            raise BreezeEngineError("Breeze worker must emit 24 kHz PCM")
        if require_native_controls and not health.get("native_checkpoints"):
            raise BreezeEngineError(
                "This Breeze worker lacks native pause/cancel checkpoints. "
                "Use the patched worker described in docs/engines/breeze.md."
            )
        if health.get("backend"):
            self.device = str(health["backend"]).lower()
        if self.reference_prefix_cache and health.get("reference_prefix_cache") is not True:
            raise BreezeEngineError(
                "Breeze worker does not advertise reference prefix cache support"
            )
        if self._owns_aligner:
            from ..alignment import StreamingTextAligner
            self.streaming_aligner = StreamingTextAligner(
                alignment_model,
                language=alignment_language,
                threads=self.alignment_threads,
            )

    def post_init(self):
        self.engine_name = "breeze"
        self.preload_sentence_tokenizer = True
        self.provides_word_timings = self.streaming_aligner is not None
        if self._initial_voice is not None:
            self.set_voice(self._initial_voice)

    def reset_audio_duration(self):
        super().reset_audio_duration()
        self._alignment_generation = getattr(self, "_alignment_generation", 0) + 1

    @staticmethod
    def alignment_non_speech_ranges(text: str) -> list[dict[str, int]]:
        return alignment_non_speech_ranges(text)

    def _alignment_language_for_voice(self) -> str | None:
        language = getattr(self._voice, "language", None)
        if language is None or str(language).casefold() == "auto":
            language = self.alignment_language
        if language is None or str(language).casefold() == "auto":
            language = getattr(getattr(self.streaming_aligner, "backend", None), "language", None)
        return language

    def get_stream_info(self):
        return 8, 1, SAMPLE_RATE  # paInt16, mono, Hz; no PyAudio import for servers

    def get_voices(self):
        response = self._http.get(self.base_url + "/v1/voices", timeout=5)
        response.raise_for_status()
        payload = response.json()
        voices = [BreezeVoice(v["id"], ref_text=v.get("ref_text"), voice_id=v["id"])
                  for v in payload]
        from ..voices import bundled_voices_data
        names = {voice.name for voice in voices}
        voices.extend(BreezeVoice(**data) for data in bundled_voices_data("breeze")
                      if data["name"] not in names)
        return voices

    def set_voice(self, voice):
        if getattr(voice, "voice_design", False):
            if voice.ref_audio or voice.ref_text or voice.voice_id:
                raise BreezeEngineError("Voice design must not include a reference voice")
            if not voice.instruct or not voice.instruct.strip():
                raise BreezeEngineError("Voice design requires a voice description")
            self._voice = voice
            self._voice_id = ""
            return
        if isinstance(voice, str):
            voice = BreezeVoice(voice, voice_id=voice)
        if getattr(voice, "spk_path", None):
            raise BreezeEngineError("Breeze needs WAV plus transcript, not Qwen latents")
        path = getattr(voice, "ref_audio", None)
        transcript = getattr(voice, "ref_text", None) or ""
        native_id = getattr(voice, "voice_id", None)
        if path is not None:
            path = Path(path).resolve(strict=True)
            if not transcript.strip():
                raise BreezeEngineError("Voice cloning requires the exact reference transcript")
            stat = path.stat()
            key = (str(path), stat.st_size, stat.st_mtime_ns, transcript)
            native_id = self._prepared.get(key)
            if native_id is None:
                data = path.read_bytes()
                native_id = "rtt_" + hashlib.sha256(
                    data + b"\0" + transcript.encode("utf-8")
                ).hexdigest()[:32]
                response = self._http.post(
                    self.base_url + "/v1/voices",
                    files={"ref_audio": (path.name, data, "audio/wav")},
                    data={"name": native_id, "ref_text": transcript},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                if response.json().get("id") != native_id:
                    raise BreezeEngineError("Worker did not register the requested voice")
                self._prepared[key] = native_id
        if not native_id:
            raise BreezeEngineError("Supply a reference WAV and transcript or a cached voice ID")
        self._voice = BreezeVoice(
            str(voice.name), path, transcript,
            getattr(voice, "instruct", None), getattr(voice, "language", "auto"),
            native_id,
        )
        self._voice_id = native_id

    def get_language(self):
        return self._voice.language

    def set_language(self, language):
        from ..language_router import normalize_qwen_language
        language = normalize_qwen_language(language, allow_auto=True)
        if language not in {"english", "chinese", "auto"}:
            raise ValueError("Breeze supports English and Chinese")
        self._voice = replace(self._voice, language=language)

    def get_instructions(self):
        return getattr(self, "_delivery_instructions", None)

    def set_instructions(self, instructions):
        if instructions is not None and not isinstance(instructions, str):
            raise TypeError("instructions must be a string or None")
        self._delivery_instructions = instructions.strip() if instructions else None

    def set_voice_parameters(self, **parameters):
        accepted = {"cfg_scale", "seed", "max_new_tokens", "temperature", "top_k",
                    "top_p", "repetition_penalty"}
        for name, value in parameters.items():
            if name not in accepted:
                raise ValueError(f"Breeze does not support parameter {name!r}")
            setattr(self, name, value)

    def create_stream_context(self):
        return _BreezeStreamContext()

    def set_stream_context(self, context):
        self._stream_context = context

    def close_stream_context(self, context):
        with self._control:
            context.closed = True
            sock, context.socket = context.socket, None
            context.started = False
            if sock is not None:
                sock.close(timeout=0)

    def _cancelled(self):
        # The server's cancellation event fences the begin_engine -> synthesize race.
        state = getattr(self.queue, "state", None)
        return self._cancel.is_set() or (
            state is not None and state.cancelled.is_set()
        )

    def _send(self, message):
        if self._socket is not None:
            context = self._stream_context
            command = {"at_monotonic": time.monotonic(), "payload": dict(message), "sent": False}
            if context is not None:
                context.commands.append(command)
            self._socket.send(json.dumps(message, ensure_ascii=False))
            command["sent"] = True

    def synthesis_diagnostics(self):
        return {
            **self.last_diagnostics,
            "model": self.model_id, "quant": self.quant, "device": self.device,
            "worker_url": self.base_url, "worker_ws_url": self.ws_url,
            "worker_health": self._worker_health,
            "reference_prefix_cache": self.reference_prefix_cache,
            "voice": {"name": self._voice.name, "native_id": self._voice_id,
                      "ref_audio": str(self._voice.ref_audio) if self._voice.ref_audio else None,
                      "ref_text": self._voice.ref_text, "voice_design": self._voice.voice_design},
        }

    def stop(self):
        self.stop_synthesis_event.set()
        self._cancel_requested_at = time.monotonic()
        self._cancel.set()
        self._pause_requested.clear()
        with self._control:
            if self._socket_started:
                try:
                    self._send({"type": "cancel"})
                except Exception:
                    if self._socket is not None:
                        self._socket.shutdown()

    def pause(self, timeout=2.0):
        self._pause_requested.set()
        self._paused.clear()
        with self._control:
            if self._idle.is_set():
                return True
            if self._socket_started:
                self._send({"type": "pause"})
        deadline = time.monotonic() + timeout
        while not self._paused.is_set():
            if self._idle.is_set() or self._cancelled():
                return True
            if time.monotonic() >= deadline:
                return False
            self._paused.wait(min(0.01, max(0, deadline - time.monotonic())))
        return True

    def resume(self):
        self._pause_requested.clear()
        self._paused.clear()
        with self._control:
            if self._socket_started:
                self._send({"type": "resume"})

    def _speak_block(
        self, text: str, instruction: str, *,
        alignment: Any = None,
        alignment_ready: Any = None,
        release_alignment: bool = True,
    ):
        if self._cancelled():
            return False
        context = self._stream_context
        signature = (self._voice_id, self.cfg_scale, self.seed, self.max_new_tokens,
                     self.temperature, self.top_k, self.top_p, self.repetition_penalty)
        with self._control:
            if context.closed:
                raise BreezeEngineError("Breeze response has already ended")
            if context.socket is not None and context.signature != signature:
                context.socket.close(timeout=0)
                context.socket, context.started = None, False
            if context.socket is None:
                context.socket = websocket.create_connection(
                    self.ws_url, timeout=0.25, enable_multithread=True)
                context.signature = signature
            sock = context.socket
            reused = context.started
        began = time.monotonic()
        record = {"text": text, "instruction": instruction, "session_reused": reused,
                  "parameters": {name: getattr(self, name) for name in (
                      "cfg_scale", "seed", "max_new_tokens", "temperature",
                      "top_k", "top_p", "repetition_penalty")}, "events": []}
        context.blocks.append(record)
        audio_hash = hashlib.sha256()
        audio_bytes = 0
        first_audio = None
        cancelled_at = None
        native_metrics = {}
        with self._control:
            self._socket = sock
            self._socket_started = context.started
        try:
            if reused:
                with self._control:
                    self._send({"type": "instruction", "instruction": instruction})
                    self._send({"type": "text", "text": text})
                    self._send({"type": "end"})
            while True:
                if self._cancelled() and cancelled_at is None:
                    cancelled_at = time.monotonic()
                    with self._control:
                        if self._socket_started:
                            self._send({"type": "cancel"})
                if cancelled_at is not None and time.monotonic() - cancelled_at > 2:
                    raise BreezeEngineError("Native cancellation was not acknowledged within 2 seconds")
                if time.monotonic() - began > self.timeout:
                    raise BreezeEngineError("Breeze synthesis timed out")
                try:
                    data = sock.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if isinstance(data, bytes):
                    if data and not self._cancelled():
                        if len(data) % 2:
                            raise BreezeEngineError("Invalid odd-sized PCM chunk")
                        if first_audio is None:
                            first_audio = time.monotonic()
                        audio_bytes += len(data)
                        audio_hash.update(data)
                        self.queue.put(data)
                        if release_alignment and alignment_ready is not None:
                            alignment_ready.set()
                        if alignment is not None:
                            audio = np.frombuffer(data, dtype="<i2").astype(np.float32)
                            alignment.feed(audio / 32768.0)
                        self.audio_duration += len(data) / (SAMPLE_RATE * 2)
                    continue
                if not data:
                    raise BreezeEngineError("Breeze closed the stream before completion")
                event = json.loads(data)
                kind = event.get("type")
                record["events"].append(event)
                if kind == "ready":
                    with self._control:
                        self._send({
                            "type": "start", "voice_id": self._voice_id,
                            "instruction": instruction, "cfg_scale": self.cfg_scale,
                            "seed": self.seed, "max_new_tokens": self.max_new_tokens,
                            "temperature": self.temperature, "top_k": self.top_k,
                            "top_p": self.top_p, "repetition_penalty": self.repetition_penalty,
                            "split_chars": 600,
                            **(
                                {"cache_reference_prefix": True}
                                if self.reference_prefix_cache else {}
                            ),
                        })
                elif kind == "started":
                    with self._control:
                        self._socket_started = context.started = True
                        if self._cancelled():
                            self._send({"type": "cancel"})
                        else:
                            if self._pause_requested.is_set():
                                self._send({"type": "pause"})
                            self._send({"type": "text", "text": text})
                            self._send({"type": "end"})
                elif kind == "metrics":
                    native_metrics = event
                elif kind == "paused":
                    self._paused.set()
                elif kind == "resumed":
                    self._paused.clear()
                elif kind in {"done", "cancelled"}:
                    elapsed = time.monotonic() - began
                    self.last_metrics = {
                        "session_reused": reused,
                        "preceding_audio_ms": native_metrics.get("preceding_frames", 0) * 80,
                        "reference_cache_hits": native_metrics.get("reference_cache_hits", 0),
                        "first_audio_ms": None if first_audio is None else (first_audio-began)*1000,
                        "audio_s": audio_bytes / (SAMPLE_RATE*2), "wall_s": elapsed,
                        "cancelled": kind == "cancelled" or self._cancelled(),
                        "native_cancel_ack_ms": None if self._cancel_requested_at is None else (time.monotonic() - self._cancel_requested_at)*1000,
                    }
                    if audio_bytes:
                        self.last_metrics["rtf"] = elapsed / self.last_metrics["audio_s"]
                    LOGGER.info("BREEZE_CONTINUITY %s", json.dumps(self.last_metrics))
                    return audio_bytes > 0 and not self._cancelled()
                elif kind == "error":
                    raise BreezeEngineError(event.get("message", "Breeze worker failed"))
        finally:
            record.update(audio_bytes=audio_bytes, pcm_sha256=audio_hash.hexdigest(),
                          cancelled=self._cancelled(), wall_s=time.monotonic() - began)
            self.last_diagnostics = {
                "blocks": [dict(block) for block in context.blocks],
                "commands": [dict(command) for command in context.commands],
            }
            with self._control:
                self._socket_started = False
                self._socket = None
            # The response owner closes the connection; a fragment ending does not.

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)
        with self._generation_lock:
            owns_context = self._stream_context is None
            if owns_context:
                self._stream_context = self.create_stream_context()
            with self._control:
                self._cancel.clear()
                self._cancel_requested_at = None
                self.stop_synthesis_event.clear()
                self._idle.clear()
            self.last_error = None
            self.last_diagnostics = {}
            try:
                if not self._voice_id and not self._voice.voice_design:
                    raise BreezeEngineError("Set a Breeze reference voice before synthesis")
                instruction = self._voice.instruct or "Speak naturally. Keep the reference speaker's voice."
                delivery = getattr(self, "_delivery_instructions", None)
                if delivery:
                    instruction = (instruction + " Delivery: " + delivery
                                   if self._voice.voice_design else delivery)
                blocks = list(directed_blocks(text, instruction))

                def make_alignment_callback(
                    timing_offset: float,
                    timing_generation: int,
                    alignment_callback: Callable[[dict[str, Any]], None] | None,
                    alignment_aborted: threading.Event,
                ):
                    # Bind all paragraph-specific state in this call frame. A
                    # previous paragraph may finish asynchronously after the
                    # next paragraph has already started. finish() is only a
                    # request to finalize; the final callback can arrive later.
                    def aligned(event: dict[str, Any]) -> None:
                        if timing_generation != self._alignment_generation:
                            return
                        event_type = event.get("type")
                        if alignment_aborted.is_set() or self.stop_synthesis_event.is_set():
                            if event_type != "alignment_done":
                                return
                        for word in event.get("words", ()):
                            self.timings.put(TimingInfo(
                                timing_offset + word["start_time"],
                                timing_offset + word["end_time"],
                                word["word"],
                            ))
                        if alignment_callback is not None:
                            try:
                                measured = bool(event.get("words") or event.get("characters"))
                                alignment_callback({**event, "engine_audio_offset": timing_offset} if measured else event)
                            except Exception:
                                LOGGER.exception("Breeze alignment callback failed")

                    return aligned

                for words, direction in blocks:
                    if self._cancelled():
                        return False
                    alignment = None
                    alignment_aborted = threading.Event()
                    alignment_ready = self.alignment_start_event or threading.Event()
                    publish_releases_alignment = self.alignment_start_event is None
                    timing_offset = self.audio_duration
                    timing_generation = self._alignment_generation
                    alignment_callback = self.on_alignment
                    target = strip_inline_vocal_tags(words).strip()
                    aligned = make_alignment_callback(
                        timing_offset, timing_generation, alignment_callback,
                        alignment_aborted,
                    )

                    self.last_alignment = None
                    if self.streaming_aligner is not None and target:
                        try:
                            alignment = self.streaming_aligner.start(
                                target, SAMPLE_RATE, callback=aligned,
                                stop_on_completion=False,
                                language=self._alignment_language_for_voice(),
                                start_after=alignment_ready,
                                cancel_event=self.alignment_cancel_event or self._cancel,
                            )
                        except Exception as exc:
                            LOGGER.warning("Breeze alignment unavailable: %s", exc)
                            aligned({
                                "type": "alignment_done", "error": str(exc),
                                "complete": False,
                            })
                    if self.streaming_aligner is not None and not target:
                        # A vocal-only fragment has no words to align, but the
                        # streaming server still needs its completion signal.
                        aligned({"type": "alignment_done", "complete": True})
                    self.last_alignment = alignment
                    try:
                        if not self._speak_block(
                            words, direction, alignment=alignment,
                            alignment_ready=alignment_ready,
                            release_alignment=publish_releases_alignment,
                        ):
                            return False
                    finally:
                        alignment_cancelled = self._cancelled() or self.stop_synthesis_event.is_set()
                        if alignment_cancelled:
                            alignment_aborted.set()
                        if alignment is not None:
                            alignment.finish(cancelled=alignment_cancelled)
                return bool(blocks)
            except Exception as exc:
                self.last_error = exc
                LOGGER.error("Breeze synthesis failed: %s", exc)
                return False
            finally:
                if owns_context:
                    self.close_stream_context(self._stream_context)
                    self._stream_context = None
                self._idle.set()
                self._paused.clear()

    def warmup(self):
        # Fill the decoder context and reserve ordinary prompt capacity before readiness.
        text = (
            "The voice is ready, and I am warming the streaming decoder so that "
            "both short replies and longer passages can be spoken smoothly, "
            "clearly, and without interruption."
        ) if self.reference_prefix_cache else "The voice is ready."
        if not self.synthesize(text):
            raise BreezeEngineError(str(self.last_error or "Breeze warmup produced no audio"))

    def shutdown(self):
        self.stop()
        self._idle.wait(3)
        if self._owns_aligner and self.streaming_aligner is not None:
            self.streaming_aligner.close()
            self.streaming_aligner = None
            self._owns_aligner = False
        self._http.close()
