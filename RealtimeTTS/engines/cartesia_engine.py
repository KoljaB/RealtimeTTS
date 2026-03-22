from __future__ import annotations

import base64
import os
import traceback
import uuid
from dataclasses import dataclass
from queue import Queue
from typing import Any, Dict, List, Mapping, Optional, Union

import pyaudio
from cartesia import Cartesia

from .base_engine import BaseEngine, TimingInfo


@dataclass
class CartesiaVoice:
    id: str
    name: Optional[str] = None
    language: Optional[str] = None

    def __repr__(self) -> str:
        if self.name:
            return f"CartesiaVoice(name={self.name}, id={self.id})"
        return f"CartesiaVoice(id={self.id})"


class CartesiaEngine(BaseEngine):
    DEFAULT_OUTPUT_FORMAT: Dict[str, Any] = {
        "container": "raw",
        "encoding": "pcm_f32le",
        "sample_rate": 44100,
    }

    VALID_MODELS = {"sonic-2", "sonic-3", "sonic-turbo", "sonic"}

    def __init__(
        self,
        api_key: str = "",
        model_id: str = "sonic-3",
        voice_id: str = "",
        output_format: Optional[Mapping[str, Any]] = None,
        language: Optional[str] = None,
        add_timestamps: bool = False,
        debug: bool = False,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        fetch_all_voices: bool = False,        
    ):
        super().__init__()

        self.debug = debug
        self.model_id = model_id
        self.language = language
        self.add_timestamps = add_timestamps
        self.timeout = timeout
        self.max_retries = max_retries

        self.output_format = self._normalize_output_format(output_format)
        self._validate_output_format(self.output_format)

        self.voice_id = ""
        self.current_voice: Optional[Union[str, CartesiaVoice]] = None
        self.set_voice(voice_id)

        self.fetch_all_voices = fetch_all_voices

        self.queue = Queue()
        self.audio_duration = 0.0

        self.api_key = api_key or os.environ.get("CARTESIA_API_KEY", "")
        if not self.api_key:
            raise ValueError("Missing Cartesia API key")

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries

        self.client = Cartesia(**client_kwargs)
        self.post_init()

    def post_init(self):
        self.engine_name = "cartesia"

    def _normalize_output_format(
        self,
        output_format: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if output_format is None:
            return dict(self.DEFAULT_OUTPUT_FORMAT)

        if isinstance(output_format, dict):
            return dict(output_format)

        if hasattr(output_format, "model_dump"):
            return dict(output_format.model_dump())

        if hasattr(output_format, "to_dict"):
            return dict(output_format.to_dict())

        return dict(output_format)

    def _validate_output_format(self, output_format: Mapping[str, Any]) -> None:
        container = str(output_format.get("container", "")).lower()
        encoding = str(output_format.get("encoding", "")).lower()

        if container != "raw":
            raise ValueError("output_format['container'] must be 'raw'")

        if encoding not in {"pcm_f32le", "pcm_s16le"}:
            raise ValueError("unsupported encoding")

        sample_rate = output_format.get("sample_rate")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("invalid sample_rate")

    def _encoding_to_pyaudio_format(self) -> int:
        encoding = str(self.output_format.get("encoding", "")).lower()

        if encoding == "pcm_f32le":
            return pyaudio.paFloat32
        if encoding == "pcm_s16le":
            return pyaudio.paInt16

        raise ValueError("unsupported encoding")

    def get_stream_info(self):
        return (
            self._encoding_to_pyaudio_format(),
            1,
            int(self.output_format["sample_rate"]),
        )

    def _voice_to_api_payload(self) -> Dict[str, Any]:
        if not self.voice_id:
            raise ValueError("No voice set")

        return {
            "mode": "id",
            "id": self.voice_id,
        }

    def _extract_audio_bytes(self, response: Any) -> bytes:
        audio = getattr(response, "audio", None)
        if audio:
            if isinstance(audio, (bytes, bytearray, memoryview)):
                return bytes(audio)
            if isinstance(audio, str):
                try:
                    return base64.b64decode(audio)
                except Exception:
                    return audio.encode("utf-8", errors="replace")

        data = getattr(response, "data", None)
        if data:
            if isinstance(data, (bytes, bytearray, memoryview)):
                return bytes(data)
            if isinstance(data, str):
                try:
                    return base64.b64decode(data)
                except Exception:
                    return data.encode("utf-8", errors="replace")

        return b""

    def _handle_timestamps(self, response: Any) -> None:
        word_timestamps = getattr(response, "word_timestamps", None)
        if not word_timestamps:
            return

        words = getattr(word_timestamps, "words", None) or []
        starts = getattr(word_timestamps, "start", None) or []
        ends = getattr(word_timestamps, "end", None) or []

        for word, start, end in zip(words, starts, ends):
            try:
                timing = TimingInfo(
                    float(start) + self.audio_duration,
                    float(end) + self.audio_duration,
                    str(word),
                )
                self.timings.put(timing)
            except Exception:
                if self.debug:
                    traceback.print_exc()

    def synthesize(self, text: str) -> bool:
        if not text:
            return True

        self.audio_duration = 0.0

        request: Dict[str, Any] = {
            "model_id": self.model_id,
            "transcript": text,
            "voice": self._voice_to_api_payload(),
            "output_format": dict(self.output_format),
        }

        if self.language:
            request["language"] = self.language

        if self.add_timestamps:
            request["add_timestamps"] = True

        request["context_id"] = uuid.uuid4().hex

        try:
            with self.client.tts.websocket_connect() as connection:
                connection.send(request)

                for response in connection:
                    response_type = getattr(response, "type", None)

                    if response_type == "chunk":
                        audio_bytes = self._extract_audio_bytes(response)
                        if audio_bytes:
                            self.queue.put(audio_bytes)

                            sample_width = 4 if self.output_format["encoding"] == "pcm_f32le" else 2
                            frames = len(audio_bytes) // sample_width
                            self.audio_duration += frames / float(self.output_format["sample_rate"])

                    elif response_type == "timestamps":
                        self._handle_timestamps(response)

                    elif response_type == "error":
                        error_msg = getattr(response, "error", "Unknown error")
                        raise RuntimeError(error_msg)

                    elif response_type == "done" or getattr(response, "done", False):
                        break

            return True

        except Exception as e:
            if self.debug:
                traceback.print_exc()
            print(f"[CartesiaEngine] Synthesis failed: {e}")
            return False

    def set_voice(self, voice: Union[str, CartesiaVoice, Dict[str, Any]]) -> None:
        if isinstance(voice, CartesiaVoice):
            self.voice_id = voice.id
            self.current_voice = voice
            return

        if isinstance(voice, dict):
            voice_id = voice.get("id") or voice.get("voice_id") or ""
            self.voice_id = str(voice_id)
            self.current_voice = voice
            return

        if isinstance(voice, str):
            voice = voice.strip()
            self.voice_id = voice
            self.current_voice = voice
            return

        raise TypeError("Unsupported voice type")

    def set_fetch_all_voices(self, enabled: bool):
        self.fetch_all_voices = bool(enabled)

    def get_voices(self) -> List[CartesiaVoice]:
        voices: List[CartesiaVoice] = []

        try:
            if self.fetch_all_voices:
                # full pagination (many requests)
                for v in self.client.voices.list():
                    voice_id = getattr(v, "id", None)
                    if not voice_id:
                        continue
                    voices.append(
                        CartesiaVoice(
                            id=str(voice_id),
                            name=getattr(v, "name", None),
                            language=getattr(v, "language", None),
                        )
                    )
            else:
                # only first page (single request)
                page = self.client.voices.list()
                for v in page.data:
                    voice_id = getattr(v, "id", None)
                    if not voice_id:
                        continue
                    voices.append(
                        CartesiaVoice(
                            id=str(voice_id),
                            name=getattr(v, "name", None),
                            language=getattr(v, "language", None),
                        )
                    )

        except Exception:
            if self.debug:
                traceback.print_exc()

        return voices

    def set_voice_parameters(self, **voice_parameters):
        if "model_id" in voice_parameters:
            self.model_id = str(voice_parameters["model_id"])

        if "voice_id" in voice_parameters:
            self.set_voice(str(voice_parameters["voice_id"]))

        if "voice" in voice_parameters:
            self.set_voice(voice_parameters["voice"])

        if "output_format" in voice_parameters:
            self.output_format = self._normalize_output_format(voice_parameters["output_format"])
            self._validate_output_format(self.output_format)

        if "language" in voice_parameters:
            self.language = voice_parameters["language"]

        if "add_timestamps" in voice_parameters:
            self.add_timestamps = bool(voice_parameters["add_timestamps"])

    def shutdown(self):
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                if self.debug:
                    traceback.print_exc()