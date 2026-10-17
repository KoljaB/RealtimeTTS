"""60db workspace voices via the authenticated REST synthesis endpoint."""

import base64
import binascii
import io
import json
import math
import os
import wave
from dataclasses import dataclass
from urllib.request import Request, urlopen

from .._audio_backend import pyaudio
from .base_engine import BaseEngine


@dataclass
class SixtyDBVoice:
    voice_id: str
    name: str = ""
    language: str = ""
    model: str = ""

    def __repr__(self):
        return self.name or self.voice_id


class SixtyDBEngine(BaseEngine):
    """Emit mono PCM16 at 24 kHz; voice IDs belong to your 60db workspace.

    NDJSON PCM is queued as it arrives. Binary WAV responses are buffered to
    validate their format before removing the container header. A failed stream
    may already have queued audio; ``synthesize`` returns False in that case.
    Cancellation waits for connection setup or the current read to finish
    (subject to ``timeout``); the synthesis thread closes its own response.
    """

    def __init__(self, voice, api_key=None, model=None, speed=1.0, timeout=60.0):
        key = api_key or os.environ.get("SIXTYDB_API_KEY")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Provide api_key or SIXTYDB_API_KEY")
        self.api_key = key
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.timeout = timeout
        self.base_url = "https://api.60db.ai"
        self.last_error = None
        self.set_voice(voice)
        self.set_voice_parameters(speed=speed, model=model)

    def post_init(self):
        self.engine_name = "sixtydb"

    def get_stream_info(self):
        return pyaudio.paInt16, 1, 24000

    def set_voice(self, voice):
        voice_id = voice.voice_id if isinstance(voice, SixtyDBVoice) else voice
        if not isinstance(voice_id, str) or not voice_id.strip():
            raise ValueError("An explicit 60db workspace voice ID is required")
        self.voice = voice_id

    def set_voice_parameters(self, **parameters):
        unknown = parameters.keys() - {"speed", "model"}
        if unknown:
            raise ValueError("Unsupported voice parameters: " + ", ".join(sorted(unknown)))
        speed = parameters.get("speed", getattr(self, "speed", 1.0))
        model = parameters.get("model", getattr(self, "model", None))
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
            raise ValueError("speed must be finite and between 0.5 and 2.0")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("model must be a nonempty model ID or None")
        self.speed, self.model = speed, model

    def _request(self, path, payload=None):
        return Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )

    @staticmethod
    def _validate_metadata(record):
        if not isinstance(record, dict):
            raise ValueError("60db returned an invalid response object")
        if record.get("success") is False or record.get("type") == "error" or record.get("error"):
            raise ValueError("60db reported a synthesis error")
        for key in ("encoding", "audio_encoding", "output_format"):
            value = record.get(key)
            if value is not None and str(value).lower() not in {"linear16", "pcm", "pcm16", "wav"}:
                raise ValueError("60db returned unsupported audio encoding")
        for key, expected in (("sample_rate", 24000), ("sample_rate_hertz", 24000), ("channels", 1), ("bit_depth", 16)):
            if key in record and record[key] != expected:
                raise ValueError("60db returned incompatible audio metadata")
        if "audio_config" in record:
            SixtyDBEngine._validate_metadata(record["audio_config"])

    @staticmethod
    def _decode_audio(value):
        if not isinstance(value, str):
            raise ValueError("60db audio must be base64 text")
        audio = base64.b64decode(value, validate=True)
        # The official SDK supports a first chunk wrapping PCM in base64 JSON.
        # Raw PCM may also start with '{'; unwrap only a valid JSON object.
        if audio.startswith(b"{"):
            try:
                inner = json.loads(audio)
            except (ValueError, UnicodeDecodeError):
                return audio
            SixtyDBEngine._validate_metadata(inner)
            result = inner.get("result", inner)
            SixtyDBEngine._validate_metadata(result)
            if "audioContent" not in result:
                raise ValueError("60db audio wrapper contains no audio")
            audio = base64.b64decode(result["audioContent"], validate=True)
        return audio

    @staticmethod
    def _pcm(audio):
        if audio.startswith(b"RIFF"):
            with wave.open(io.BytesIO(audio), "rb") as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 24000, "NONE"):
                    raise ValueError("60db WAV must be mono PCM16 at 24000 Hz")
                frames = wav.getnframes()
                audio = wav.readframes(frames)
                if len(audio) != frames * 2:
                    raise ValueError("60db returned truncated WAV audio")
        elif audio.startswith((b"ID3", b"OggS", b"fLaC")):
            raise ValueError("60db returned compressed audio instead of PCM")
        if len(audio) % 2:
            raise ValueError("60db returned an incomplete PCM16 sample")
        return audio

    def _enqueue(self, audio):
        audio = self._pcm(audio)
        if audio and not self.stop_synthesis_event.is_set():
            self.queue.put(audio)
            self.audio_duration += len(audio) / 48000
            return True
        return False

    def _record_audio(self, record):
        self._validate_metadata(record)
        result = record.get("result", record.get("backendResponse", record))
        self._validate_metadata(result)
        value = result.get("audioContent", result.get("audio_base64"))
        return self._decode_audio(value) if value is not None else b""

    def synthesize(self, text, sentence_count=0):
        super().synthesize(text, sentence_count)
        self.last_error = None
        if not isinstance(text, str) or not text.strip() or len(text) > 5000:
            self.last_error = ValueError("text must contain 1 to 5000 characters")
            return False
        payload = {
            "text": text, "voice_id": self.voice, "speed": self.speed,
            "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 24000},
            "output_format": "wav", "timestamp_type": "NONE",
        }
        if self.model is not None:
            payload["model_id"] = self.model
        produced = False
        try:
            with urlopen(self._request("/tts-synthesize", payload), timeout=self.timeout) as response:
                self._validate_metadata({
                    key: int(response.headers[header])
                    for key, header in (("sample_rate", "X-Sample-Rate"), ("channels", "X-Channels"), ("bit_depth", "X-Bit-Depth"))
                    if header in response.headers
                })
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                if content_type.startswith("audio/") or content_type == "application/octet-stream":
                    if content_type not in {"audio/wav", "audio/x-wav", "audio/pcm", "audio/octet-stream", "application/octet-stream"}:
                        raise ValueError("60db returned unsupported audio content type")
                    audio = bytearray()
                    while not self.stop_synthesis_event.is_set():
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        audio.extend(chunk)
                    produced = self._enqueue(bytes(audio))
                elif content_type == "application/json":
                    produced = self._enqueue(self._record_audio(json.load(response)))
                elif content_type in {"application/x-ndjson", "application/ndjson", "text/plain"}:
                    for line in response:
                        if self.stop_synthesis_event.is_set():
                            break
                        if line.strip():
                            produced = self._enqueue(self._record_audio(json.loads(line))) or produced
                else:
                    raise ValueError("60db returned unsupported response content type")
            if self.stop_synthesis_event.is_set():
                return False
            if not produced:
                raise ValueError("60db produced no audio")
            return True
        except (OSError, ValueError, TypeError, KeyError, EOFError, wave.Error, binascii.Error) as exc:
            self.last_error = exc
            return False
    def get_voices(self, model="quality"):
        """Fetch actual workspace voices in the quality or fast tier."""
        if model not in {"quality", "fast"}:
            raise ValueError("Voice model tier must be quality or fast")
        with urlopen(self._request("/voices?model=" + model), timeout=self.timeout) as response:
            record = json.load(response)
        self._validate_metadata(record)
        voices = record.get("data")
        if not isinstance(voices, list):
            raise ValueError("60db returned an invalid voice list")
        result = []
        for voice in voices:
            voice_id = voice["voice_id"]
            if not isinstance(voice_id, str) or not voice_id.strip():
                raise ValueError("60db returned an invalid voice ID")
            result.append(SixtyDBVoice(voice_id, voice.get("name", ""), (voice.get("labels") or {}).get("language", ""), voice.get("model", "")))
        return result

    def shutdown(self):
        self.stop()
