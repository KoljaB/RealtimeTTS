"""Direct streaming client for the RealtimeTTS Qwen HTTP server."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Mapping

import requests

from .._audio_formats import INT16
from .base_engine import BaseEngine


SAMPLE_RATE = 24_000
_FRAME_BYTES = 2
_AUDIO_TYPES = {"audio/pcm", "application/octet-stream"}


@dataclass(frozen=True, slots=True)
class QwenServerVoice:
    name: str
    kind: str = "server"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("QwenServerVoice.name must not be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", str(self.kind or "server"))


def _voice_name(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("name")
    elif not isinstance(value, str):
        value = getattr(value, "name", value)
    name = str(value).strip() if value is not None else ""
    if not name:
        raise ValueError("Qwen server voice must have a non-empty name")
    return name


def _timeout_value(value: Any) -> float | tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("timeout must be a positive number or (connect, read)")
        result = (float(value[0]), float(value[1]))
    else:
        result = float(value)
    values = result if isinstance(result, tuple) else (result,)
    if any(not math.isfinite(item) or item <= 0 for item in values):
        raise ValueError("timeout values must be finite and positive")
    return result


class QwenServerEngine(BaseEngine):
    """Stream raw mono 24 kHz s16le from a Qwen HTTP server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18085",
        voice: str = "alex",
        *,
        api_key: str | None = None,
        language: str = "english",
        seed: int = -1,
        timeout: float | tuple[float, float] = (5, 30),
        clone_mode: str = "speaker_only",
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        self.voice = _voice_name(voice)
        self.current_voice = QwenServerVoice(self.voice)
        self.language = str(language).strip().lower()
        if not self.language:
            raise ValueError("language must not be empty")
        self.seed = self._validate_seed(seed)
        self.timeout = _timeout_value(timeout)
        self.clone_mode = self._validate_clone_mode(clone_mode)
        self.max_new_tokens = self._validate_max_new_tokens(max_new_tokens)
        self.temperature = self._validate_temperature(temperature)
        self._api_key = str(api_key).strip() if api_key else None
        self.session = requests.Session()
        self._active_response: Any = None
        self._response_lock = threading.RLock()
        self._shutdown = False
        self.last_error: BaseException | None = None
        self.last_first_audio_time: float | None = None
        self.first_audio_time: float | None = None
        self.last_first_audio_ms: float | None = None
        self.last_wall_time: float | None = None
        self.last_chunk_count = 0
        self.last_synthesis_profile: dict[str, Any] = {}

    def post_init(self) -> None:
        self.engine_name = "qwen-server"
        self.can_consume_generators = False

    @staticmethod
    def _validate_seed(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("seed must be an integer")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer") from exc
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("seed must be an integer")
        return result

    @staticmethod
    def _validate_max_new_tokens(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("max_new_tokens must be positive")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_new_tokens must be positive") from exc
        if result <= 0 or (isinstance(value, float) and not value.is_integer()):
            raise ValueError("max_new_tokens must be positive")
        return result

    @staticmethod
    def _validate_temperature(value: Any) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("temperature must be finite and non-negative") from exc
        if not math.isfinite(result) or result < 0:
            raise ValueError("temperature must be finite and non-negative")
        return result

    @staticmethod
    def _validate_clone_mode(value: Any) -> str:
        result = str(value).strip()
        if result not in {"auto", "speaker_only"}:
            raise ValueError("clone_mode must be 'auto' or 'speaker_only'")
        return result

    def get_stream_info(self) -> tuple[int, int, int]:
        return INT16, 1, SAMPLE_RATE

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "audio/pcm", "Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(self, text: str) -> dict[str, Any]:
        return {
            "input": text,
            "voice": self.voice,
            "language": self.language,
            "clone_mode": self.clone_mode,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "response_format": "pcm",
        }

    @staticmethod
    def _validate_response_headers(response: Any) -> None:
        headers = {str(key).lower(): value for key, value in (response.headers or {}).items()}
        content_type = str(headers.get("content-type", "")).split(";", 1)[0].strip().lower()
        if content_type and content_type not in _AUDIO_TYPES:
            raise ValueError(f"Qwen PCM response has invalid content type {content_type!r}")
        expected = {
            "x-audio-encoding": "s16le",
            "x-audio-sample-rate": "24000",
            "x-audio-channels": "1",
            "x-audio-bits-per-sample": "16",
        }
        for name, configured in expected.items():
            if name not in headers:
                continue
            actual = str(headers[name]).strip()
            if name == "x-audio-encoding":
                valid = actual.lower() == configured
            else:
                try:
                    valid = int(actual) == int(configured)
                except (TypeError, ValueError):
                    valid = False
            if not valid:
                raise ValueError(
                    f"Qwen PCM response header {name}={actual!r} does not match {configured!r}"
                )

    @staticmethod
    def _close_response(response: Any) -> None:
        if response is None:
            return
        try:
            response.close()
        except Exception:
            pass

    def _close_active_response(self) -> None:
        with self._response_lock:
            response = self._active_response
        self._close_response(response)

    def _queue_pcm(self, payload: bytes, started_at: float) -> None:
        with self._response_lock:
            if self._shutdown or self.stop_synthesis_event.is_set():
                return
            if self.last_first_audio_time is None:
                elapsed = time.perf_counter() - started_at
                self.last_first_audio_time = elapsed
                self.first_audio_time = elapsed
                self.last_first_audio_ms = elapsed * 1000.0
            self.queue.put(payload)
            self.last_chunk_count += 1
            self.audio_duration += len(payload) / (SAMPLE_RATE * _FRAME_BYTES)

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)
        text = str(text or "").strip()
        self.last_error = None
        self.last_first_audio_time = None
        self.first_audio_time = None
        self.last_first_audio_ms = None
        self.last_wall_time = None
        self.last_chunk_count = 0
        self.last_synthesis_profile = {}
        self.audio_duration = 0.0
        started_at = time.perf_counter()
        response = None
        pending = b""
        try:
            with self._response_lock:
                if self._shutdown:
                    raise RuntimeError("QwenServerEngine is shut down")
            if not text:
                raise ValueError("QwenServerEngine text must not be empty")
            response = self.session.post(
                self.base_url + "/v1/audio/speech",
                json=self._payload(text),
                headers=self._headers(),
                stream=True,
                timeout=self.timeout,
            )
            with self._response_lock:
                abort_before_read = self._shutdown or self.stop_synthesis_event.is_set()
                if not abort_before_read:
                    self._active_response = response
            if abort_before_read:
                self._close_response(response)
                return True
            response.raise_for_status()
            self._validate_response_headers(response)
            for chunk in response.iter_content(chunk_size=None):
                if self.stop_synthesis_event.is_set():
                    break
                if not chunk:
                    continue
                pending += bytes(chunk)
                aligned = len(pending) - (len(pending) % _FRAME_BYTES)
                if aligned:
                    self._queue_pcm(pending[:aligned], started_at)
                    pending = pending[aligned:]
            if pending and not self.stop_synthesis_event.is_set():
                raise ValueError("Qwen PCM response ended with a partial audio frame")
            if self.stop_synthesis_event.is_set():
                return True
            if not self.last_chunk_count:
                raise RuntimeError("Qwen server produced no audio")
            return True
        except Exception as exc:
            if self.stop_synthesis_event.is_set() or self._shutdown:
                self.last_error = None
                return True
            self.last_error = exc
            return False
        finally:
            self._close_response(response)
            with self._response_lock:
                if self._active_response is response:
                    self._active_response = None
            self.last_wall_time = time.perf_counter() - started_at
            self.last_synthesis_profile = {
                "first_audio_ms": self.last_first_audio_ms,
                "wall_ms": self.last_wall_time * 1000.0,
                "chunk_count": self.last_chunk_count,
                "audio_bytes": int(self.audio_duration * SAMPLE_RATE * _FRAME_BYTES),
                "cancelled": self.stop_synthesis_event.is_set(),
            }

    def get_voices(self) -> list[QwenServerVoice]:
        with self._response_lock:
            if self._shutdown:
                raise RuntimeError("QwenServerEngine is shut down")
        response = self.session.get(
            self.base_url + "/v1/audio/voices",
            headers=self._headers(),
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
            payload = response.json()
        finally:
            self._close_response(response)
        values = payload.get("voices", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(values, list):
            raise ValueError("Qwen voice response must contain a voices list")
        result = []
        for value in values:
            if isinstance(value, Mapping):
                result.append(QwenServerVoice(value.get("name", ""), value.get("kind", "server")))
            else:
                result.append(QwenServerVoice(value))
        return result

    def set_voice(self, voice: Any) -> None:
        name = _voice_name(voice)
        self.voice = name
        self.current_voice = QwenServerVoice(name)

    def set_language(self, language: str) -> None:
        self.set_voice_parameters(language=language)

    def set_voice_parameters(self, **voice_parameters: Any) -> None:
        supported = {"voice", "language", "seed", "max_new_tokens", "temperature", "clone_mode"}
        unknown = set(voice_parameters) - supported
        if unknown:
            raise ValueError(f"Unsupported QwenServerEngine parameters: {sorted(unknown)}")
        if "voice" in voice_parameters:
            self.set_voice(voice_parameters.pop("voice"))
        if "language" in voice_parameters:
            language = str(voice_parameters.pop("language")).strip().lower()
            if not language:
                raise ValueError("language must not be empty")
            self.language = language
        if "seed" in voice_parameters:
            self.seed = self._validate_seed(voice_parameters.pop("seed"))
        if "max_new_tokens" in voice_parameters:
            self.max_new_tokens = self._validate_max_new_tokens(
                voice_parameters.pop("max_new_tokens")
            )
        if "temperature" in voice_parameters:
            self.temperature = self._validate_temperature(voice_parameters.pop("temperature"))
        if "clone_mode" in voice_parameters:
            self.clone_mode = self._validate_clone_mode(voice_parameters.pop("clone_mode"))

    def stop(self) -> None:
        with self._response_lock:
            super().stop()
            response = self._active_response
        self._close_response(response)

    def shutdown(self) -> None:
        with self._response_lock:
            if self._shutdown:
                return
            self._shutdown = True
            super().stop()
            response = self._active_response
        self._close_response(response)
        self.session.close()


__all__ = ["QwenServerEngine", "QwenServerVoice"]
