"""GMI Cloud text-to-speech engine for RealtimeTTS.

Uses GMI Cloud's multimodal inference request queue (one API, one auth) with
family-aware request mapping for the speech models it hosts:

- Inworld Realtime TTS (``inworld-tts-*``)
- MiniMax speech synthesis (``minimax-tts-speech-*``)
- Resemble AI's Chatterbox (``Chatterbox-tts``, requires a Resemble
  ``voice_uuid`` as ``voice``)

Both documented response shapes are resolved: synchronous inline ``outcome``
responses, and queued/dispatched generations resolved by polling
``GET /api/v1/ie/requestqueue/apikey/requests/{request_id}``.

Output contract: every response is decoded and normalized before playback to
mono 16-bit PCM at the engine's declared ``sample_rate`` (48 kHz by default),
so multi-channel responses (e.g. MiniMax's native 32 kHz stereo) are
down-mixed and resampled to match ``get_stream_info()`` exactly.

Verified live against the GMI Cloud API: synthesis on the Inworld and MiniMax
families (including normalization of MiniMax's native 32 kHz stereo output),
end-to-end playback through ``TextToAudioStream``, Chatterbox ``voice_uuid``
validation, and bad-key handling.
"""

import io
import logging
import os
import struct
import time
import wave
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import requests
from .base_engine import BaseEngine

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://console.gmicloud.ai"
QUEUE_PATH = "/api/v1/ie/requestqueue/apikey/requests"
DEFAULT_MODEL = "inworld-tts-2"
DEFAULT_SAMPLE_RATE = 48000

# Per-family request defaults proven against the live API.
FAMILY_DEFAULTS = {
    "inworld": {"voice": "Dennis"},
    "minimax": {"voice": "English_expressive_narrator"},
    "chatterbox": {"voice": None},
}

# HTTP statuses worth retrying while polling a queued generation.
TRANSIENT_POLL_STATUSES = (408, 429)


class GmicloudAPIError(RuntimeError):
    """A GMI Cloud HTTP or generation failure."""

    def __init__(self, message: str, *, status_code: int | None = None,
                 request_id: str | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.body = body


def model_family(model: str) -> str:
    """Map a GMI Cloud model id to its request family.

    Returns one of ``"inworld"``, ``"minimax"`` or ``"chatterbox"``.
    Unknown ids default to the Inworld request shape.
    """
    m = (model or "").lower()
    if m.startswith("minimax"):
        return "minimax"
    if m.startswith("chatterbox"):
        return "chatterbox"
    return "inworld"


@dataclass
class _TTSOptions:
    api_key: str
    base_url: str
    model: str
    voice: str | None
    audio_format: str
    speed: float
    temperature: float | None
    timestamp_type: str | None
    extra_payload: dict[str, Any] = field(default_factory=dict)
    poll_interval_secs: float = 2.0
    poll_attempts: int = 90


def build_payload(text: str, opts: _TTSOptions) -> dict[str, Any]:
    """Build the GMI Cloud request payload for the model's request family.

    Raises:
        ValueError: Chatterbox models without a Resemble ``voice_uuid``.
    """
    family = model_family(opts.model)
    defaults = FAMILY_DEFAULTS[family]
    voice = opts.voice if opts.voice is not None else defaults["voice"]
    encoding = (opts.audio_format or "WAV").lower()
    speed = opts.speed

    if family == "inworld":
        payload: dict[str, Any] = {
            "text": text,
            "audio_encoding": encoding.upper(),
            "speaking_rate": speed,
        }
        if voice:
            payload["voice_id"] = voice
        if opts.temperature is not None:
            payload["temperature"] = opts.temperature
        if opts.timestamp_type is not None:
            payload["timestamp_type"] = opts.timestamp_type
    elif family == "minimax":
        payload = {
            "text": text,
            "speed": str(speed),
            "format": "wav" if encoding == "wav" else encoding,
        }
        if voice:
            payload["voice_id"] = voice
    else:  # chatterbox
        if not voice:
            raise ValueError(
                "Chatterbox models require a Resemble voice UUID (voice_uuid): "
                "pass voice='<your-voice-uuid>' to GmicloudEngine "
                "(or use a different model)."
            )
        payload = {
            "data": text,
            "voice_uuid": voice,
            "output_format": "wav" if encoding == "wav" else encoding,
        }

    if opts.extra_payload:
        payload.update(opts.extra_payload)
    return payload


def _extract_audio_url(data: dict[str, Any]) -> str | None:
    """Extract the first audio URL from a queue response outcome."""
    outcome = data.get("outcome") or {}
    if not isinstance(outcome, dict):
        return None
    audio_url = outcome.get("audio_url")
    if audio_url:
        return audio_url
    for key in ("media_urls", "medias", "media"):
        for item in outcome.get(key) or []:
            if isinstance(item, dict) and item.get("url"):
                return item["url"]
    return None


def _parse_wav(data: bytes) -> tuple[bytes, int, int, int] | None:
    """Decode a WAV blob into (pcm, sample_rate, num_channels, sample_width).

    Returns ``None`` when the blob is not a parseable WAV file.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            pcm = wav.readframes(wav.getnframes())
            return pcm, wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
    except Exception:  # noqa: BLE001 - any unparsable blob is simply not a WAV
        return None


def _to_mono_s16(pcm: bytes, channels: int) -> bytes:
    """Downmix interleaved 16-bit PCM to mono by averaging the channels."""
    if channels <= 1:
        return pcm
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    mono = [sum(samples[i : i + channels]) // channels for i in range(0, len(samples), channels)]
    return struct.pack(f"<{len(mono)}h", *mono)


def _resample_s16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample mono 16-bit PCM to another rate (linear interpolation)."""
    if src_rate == dst_rate:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return pcm
    n_out = max(1, round(samples.size * dst_rate / src_rate))
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    out = np.interp(x_new, x_old, samples)
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()


class GmicloudVoice:
    """A GMI Cloud voice preset (family-specific; arbitrary ids are allowed)."""

    def __init__(self, name: str, family: str = "inworld", language: str = "english"):
        self.name = name
        self.family = family
        self.language = language

    def __repr__(self):
        return f"GmicloudVoice(name={self.name}, family={self.family}, language={self.language})"


class GmicloudEngine(BaseEngine):
    """GMI Cloud text-to-speech engine for RealtimeTTS.

    One engine covers GMI Cloud's speech model families behind the
    request-queue API. The request shape is selected from the model id and
    responses (synchronous or queued) are resolved to WAV audio, then
    normalized to mono 16-bit PCM at ``sample_rate`` before playback.

    Example::

        from RealtimeTTS import TextToAudioStream, GmicloudEngine

        engine = GmicloudEngine(api_key="...")  # or GMI_CLOUD_API_KEY env var
        stream = TextToAudioStream(engine)
        stream.feed("Hello from GMI Cloud.").play()
    """

    # Documented default voices per family. Arbitrary voice ids (including a
    # Resemble voice UUID for Chatterbox) are accepted by ``set_voice``.
    VOICES: ClassVar[list[dict[str, str]]] = [
        {"name": "Dennis", "family": "inworld", "language": "english"},
        {"name": "English_expressive_narrator", "family": "minimax", "language": "english"},
    ]

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        voice: str | None = None,
        speed: float = 1.0,
        temperature: float | None = None,
        extra_payload: dict[str, Any] | None = None,
        base_url: str = DEFAULT_BASE_URL,
        poll_interval_secs: float = 2.0,
        poll_attempts: int = 90,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        timeout: float = 60.0,
        debug: bool = False,
    ):
        """Initialize a GMI Cloud realtime text-to-speech engine.

        Args:
            api_key: GMI Cloud API key. Falls back to the ``GMI_CLOUD_API_KEY``
                or ``GMI_API_KEY`` environment variables.
            model: GMI Cloud model id. Families are selected from the id:
                ``inworld-tts-*`` (low latency), ``minimax-tts-speech-*``,
                ``Chatterbox-tts`` (requires a Resemble voice UUID as
                ``voice``). Unknown ids use the Inworld request shape.
            voice: Voice id for the selected family (defaults per family).
                Chatterbox models require a Resemble voice UUID here.
            speed: Speaking rate (1.0 = normal).
            temperature: Optional generation temperature (Inworld family).
            extra_payload: Extra fields merged into the request payload
                (merged last, so they override built-in fields).
            base_url: GMI Cloud console API base URL.
            poll_interval_secs: Delay between polls of a queued generation.
            poll_attempts: Polls before giving up on a queued generation.
                MiniMax generations go through GMI's queue and can be slow;
                raise this if long generations time out.
            sample_rate: Emitted PCM rate; responses are resampled to it.
            timeout: Per-request HTTP timeout in seconds.
            debug: Print synthesis timing and errors to stdout.
        """
        self.api_key = (
            api_key
            or os.environ.get("GMI_CLOUD_API_KEY")
            or os.environ.get("GMI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "GMI Cloud API key is required. Provide it via the api_key "
                "parameter or the GMI_CLOUD_API_KEY environment variable."
            )

        self.model = model
        self.voice = voice
        self.speed = speed
        self.temperature = temperature
        self.extra_payload = dict(extra_payload or {})
        self.base_url = base_url.rstrip("/")
        self.poll_interval_secs = poll_interval_secs
        self.poll_attempts = poll_attempts
        self.sample_rate = sample_rate
        self.timeout = timeout
        self.debug = debug
        self.queue = None

        if model_family(self.model) == "chatterbox" and not self.voice:
            raise ValueError(
                "Chatterbox models require a Resemble voice UUID (voice_uuid): "
                "pass voice='<your-voice-uuid>' to GmicloudEngine "
                "(or use a different model)."
            )

    def post_init(self):
        self.engine_name = "gmicloud"

    def get_stream_info(self):
        """PyAudio stream configuration: mono 16-bit PCM at ``sample_rate``.

        Returns:
            tuple: (format, channels, sample_rate)
        """
        import pyaudio

        return pyaudio.paInt16, 1, self.sample_rate

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        """Synthesize text to audio and push PCM into the playback queue.

        Args:
            text: Text to synthesize.
            sentence_count: Sentences synthesized so far (progress tracking).

        Returns:
            bool: True if audio was produced, False otherwise.
        """
        super().synthesize(text, sentence_count)

        if not text or not text.strip():
            return True

        if self.debug:
            start = time.time()

        try:
            wav_bytes = self._request_audio(text)
        except ValueError:
            raise  # configuration errors are the caller's to fix
        except Exception as e:  # noqa: BLE001 - engine convention: any request
            # failure is logged and reported as a failed synthesis, never raised
            logger.error("GMI Cloud TTS error: %s", e)
            if self.debug:
                print(f"GmicloudEngine request error: {e}")
            return False

        if self.stop_synthesis_event.is_set():
            return False

        parsed = _parse_wav(wav_bytes)
        if parsed is None:
            logger.error("GMI Cloud TTS: response is not WAV audio")
            return False
        pcm, rate, channels, width = parsed
        if width != 2:
            logger.error("GMI Cloud TTS: unsupported sample width %s", width)
            return False

        if channels > 1:
            pcm = _to_mono_s16(pcm, channels)
        if rate != self.sample_rate:
            pcm = _resample_s16(pcm, rate, self.sample_rate)

        if self.debug:
            print(
                f"GmicloudEngine synthesized {len(pcm) // 2} samples "
                f"in {time.time() - start:.2f}s ({rate} Hz {channels}ch -> "
                f"{self.sample_rate} Hz mono)"
            )

        self.queue.put(pcm)
        return True

    # -- request layer -----------------------------------------------------

    def _options(self) -> _TTSOptions:
        return _TTSOptions(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            voice=self.voice,
            audio_format="WAV",
            speed=self.speed,
            temperature=self.temperature,
            timestamp_type=None,
            extra_payload=self.extra_payload,
            poll_interval_secs=self.poll_interval_secs,
            poll_attempts=self.poll_attempts,
        )

    def _request_audio(self, text: str) -> bytes:
        """Run one generation to completion and return WAV bytes.

        Raises:
            ValueError: invalid configuration (e.g. Chatterbox without voice).
            GmicloudAPIError: HTTP or generation failure.
        """
        opts = self._options()
        body = {"model": opts.model, "payload": build_payload(text, opts)}
        headers = {
            "Authorization": f"Bearer {opts.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{opts.base_url}{QUEUE_PATH}",
            json=body,
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise GmicloudAPIError(
                f"GMI Cloud API error: HTTP {response.status_code} - {response.text}",
                status_code=response.status_code,
                body=response.text,
            )
        data = response.json()

        data, audio_url = self._resolve_result(opts, data)
        if data.get("status") != "success" or not audio_url:
            raise GmicloudAPIError(
                f"GMI Cloud TTS failed: status={data.get('status')!r}, "
                f"request_id={data.get('request_id')!r}",
                request_id=str(data.get("request_id") or ""),
            )

        audio = requests.get(audio_url, timeout=self.timeout)
        if audio.status_code != 200:
            raise GmicloudAPIError(
                f"GMI Cloud audio download error: HTTP {audio.status_code}",
                status_code=audio.status_code,
            )
        return audio.content

    def _resolve_result(
        self, opts: _TTSOptions, data: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        """Resolve a response that may be synchronous or queued.

        Returns the final response dict and an audio URL (or None). Permanent
        HTTP errors while polling raise immediately; only 408/429/5xx retry.
        """
        status = data.get("status")
        request_id = data.get("request_id")
        audio_url = _extract_audio_url(data)

        if not audio_url and status != "failed" and request_id:
            poll_url = f"{opts.base_url}{QUEUE_PATH}/{request_id}"
            for _ in range(opts.poll_attempts):
                time.sleep(opts.poll_interval_secs)
                response = requests.get(
                    poll_url,
                    headers={"Authorization": f"Bearer {opts.api_key}"},
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                elif (
                    response.status_code in TRANSIENT_POLL_STATUSES
                    or response.status_code >= 500
                ):
                    continue  # transient: keep polling
                else:
                    raise GmicloudAPIError(
                        f"GMI Cloud queue poll error: HTTP {response.status_code} "
                        f"- {response.text}",
                        status_code=response.status_code,
                        request_id=str(request_id),
                        body=response.text,
                    )
                status = data.get("status")
                audio_url = _extract_audio_url(data)
                if status in ("success", "failed") or audio_url:
                    break

        return data, audio_url

    # -- voice management --------------------------------------------------

    def get_voices(self):
        """Return the documented default voices (arbitrary ids also work)."""
        return [GmicloudVoice(**v) for v in self.VOICES]

    def set_voice(self, voice: str | GmicloudVoice):
        """Set the voice id for the selected model family."""
        if isinstance(voice, GmicloudVoice):
            self.voice = voice.name
        else:
            self.voice = voice

    def set_voice_parameters(self, **voice_parameters):
        """Set voice parameters: speed, model, temperature, voice."""
        if "speed" in voice_parameters:
            self.speed = voice_parameters["speed"]
        if "model" in voice_parameters:
            self.model = voice_parameters["model"]
        if "temperature" in voice_parameters:
            self.temperature = voice_parameters["temperature"]
        if "voice" in voice_parameters:
            self.set_voice(voice_parameters["voice"])

    def shutdown(self):
        """Release resources (nothing held between syntheses)."""
