"""
ModelsLab Text-to-Speech Engine for RealtimeTTS.

Uses the ModelsLab TTS API (https://modelslab.com) to synthesize speech.
Supports multiple voice IDs and languages with async polling for audio generation.

Usage:
    from RealtimeTTS import TextToAudioStream
    from RealtimeTTS.engines import ModelsLabEngine

    engine = ModelsLabEngine(api_key="YOUR_MODELSLAB_API_KEY", voice_id=1)
    stream = TextToAudioStream(engine)
    stream.feed("Hello from ModelsLab!").play()
"""
import io
import logging
import os
import time

import pyaudio
import requests

from .base_engine import BaseEngine

logger = logging.getLogger(__name__)

MODELSLAB_TTS_URL = "https://modelslab.com/api/v6/voice/text_to_speech"
MODELSLAB_FETCH_URL = "https://modelslab.com/api/v6/voice/fetch/{request_id}"
POLL_INTERVAL = 5       # seconds between polls
POLL_TIMEOUT = 300      # maximum seconds to wait for audio
CHUNK_SIZE = 4096       # bytes per audio chunk pushed to queue


class ModelsLabVoice:
    """Represents a ModelsLab TTS voice."""

    def __init__(self, voice_id: int, name: str, language: str = "english"):
        self.voice_id = voice_id
        self.name = name
        self.language = language

    def __repr__(self):
        return f"{self.name} (voice_id={self.voice_id}, language={self.language})"


# Built-in voices. More voices may be available via the ModelsLab dashboard.
MODELSLAB_VOICES = [
    ModelsLabVoice(1, "Neutral", "english"),
    ModelsLabVoice(2, "Male", "english"),
    ModelsLabVoice(3, "Warm Male", "english"),
    ModelsLabVoice(4, "Deep Male", "english"),
    ModelsLabVoice(5, "Female", "english"),
    ModelsLabVoice(6, "Clear Female", "english"),
    ModelsLabVoice(7, "Voice 7", "english"),
    ModelsLabVoice(8, "Voice 8", "english"),
    ModelsLabVoice(9, "Voice 9", "english"),
    ModelsLabVoice(10, "Voice 10", "english"),
]


class ModelsLabEngine(BaseEngine):
    """
    RealtimeTTS engine for ModelsLab TTS API.

    Authentication:
        ModelsLab uses key-in-body authentication. Pass your API key as
        ``api_key`` parameter or set the ``MODELSLAB_API_KEY`` environment variable.

    Async polling:
        ModelsLab sometimes returns ``status: "processing"`` and requires polling
        ``/voice/fetch/{request_id}`` until the audio URL is ready. This engine
        handles polling automatically.
    """

    def __init__(
        self,
        api_key: str = "",
        voice_id: int = 1,
        language: str = "english",
        speed: float = 1.0,
    ):
        """
        Initializes the ModelsLab TTS engine.

        Args:
            api_key (str): ModelsLab API key. Falls back to MODELSLAB_API_KEY env var.
            voice_id (int): Voice ID to use (1–10). Defaults to 1 (Neutral).
            language (str): Language for synthesis. Defaults to "english".
            speed (float): Speech speed (0.5–2.0). Defaults to 1.0.
        """
        self.voice_id = voice_id
        self.language = language
        self.speed = speed

        if not api_key:
            api_key = os.environ.get("MODELSLAB_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Missing ModelsLab API key. Either:\n"
                "1. Pass the API key as a parameter: ModelsLabEngine(api_key='...')\n"
                "2. Set MODELSLAB_API_KEY environment variable"
            )

        self.api_key = api_key

    def post_init(self):
        """Set engine name. ModelsLab synthesizes whole strings, not generators."""
        self.can_consume_generators = False
        self.engine_name = "modelslab"

    def get_stream_info(self):
        """
        Audio stream configuration for PyAudio.

        ModelsLab returns MP3 audio. Use paCustomFormat to stream compressed audio.

        Returns:
            tuple: (format, channels, sample_rate) — paCustomFormat, -1, -1 for MP3.
        """
        return pyaudio.paCustomFormat, -1, -1

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text to audio and pushes chunks to the audio queue.

        Args:
            text (str): Text to convert to speech.

        Returns:
            bool: True on success, False on failure.
        """
        super().synthesize(text)

        try:
            audio_url = self._get_audio_url(text)
            if not audio_url:
                return False

            # Stream the audio file in chunks
            response = requests.get(audio_url, stream=True, timeout=30)
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    self.queue.put(chunk)

            return True

        except Exception as e:
            logger.error(f"ModelsLab synthesis error: {e}")
            return False

    def _get_audio_url(self, text: str) -> str:
        """
        Calls the ModelsLab TTS API and returns the audio URL.
        Handles async polling if status is 'processing'.
        """
        body = {
            "key": self.api_key,
            "prompt": text,
            "language": self.language,
            "voice_id": self.voice_id,
            "speed": self.speed,
        }

        resp = requests.post(
            MODELSLAB_TTS_URL,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")

        if status == "error":
            logger.error(f"ModelsLab TTS error: {data.get('message', 'Unknown error')}")
            return ""

        if status == "processing":
            request_id = str(data.get("request_id", ""))
            if not request_id:
                logger.error("ModelsLab TTS returned processing without request_id")
                return ""
            return self._poll_until_ready(request_id)

        # status == "success"
        output = data.get("output", "")
        return output if isinstance(output, str) else (output[0] if output else "")

    def _poll_until_ready(self, request_id: str) -> str:
        """Poll /voice/fetch/{request_id} until the audio URL is ready."""
        fetch_url = MODELSLAB_FETCH_URL.format(request_id=request_id)
        fetch_body = {"key": self.api_key}
        deadline = time.time() + POLL_TIMEOUT

        while time.time() < deadline:
            time.sleep(POLL_INTERVAL)
            try:
                resp = requests.post(
                    fetch_url,
                    json=fetch_body,
                    headers={"Content-Type": "application/json"},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") == "success":
                    output = data.get("output", "")
                    return output if isinstance(output, str) else (output[0] if output else "")
                if data.get("status") == "error":
                    logger.error(f"ModelsLab TTS failed: {data.get('message', 'Unknown error')}")
                    return ""
            except Exception as e:
                logger.warning(f"ModelsLab poll error: {e}")

        logger.error(f"ModelsLab TTS timed out after {POLL_TIMEOUT}s (request_id={request_id})")
        return ""

    def get_voices(self) -> list:
        """Returns the list of built-in ModelsLab voices."""
        return MODELSLAB_VOICES

    def set_voice(self, voice) -> None:
        """
        Sets the active voice.

        Args:
            voice: A ModelsLabVoice object or voice_id integer.
        """
        if isinstance(voice, ModelsLabVoice):
            self.voice_id = voice.voice_id
            self.language = voice.language
            logger.info(f"Voice set to {voice}")
        elif isinstance(voice, int):
            self.voice_id = voice
            logger.info(f"Voice ID set to {voice}")
        else:
            logger.warning(f"Unknown voice type: {type(voice)}")

    def set_voice_parameters(self, **kwargs) -> None:
        """
        Sets voice parameters.

        Keyword Args:
            speed (float): Speech speed 0.5–2.0.
            language (str): Target language (e.g. 'english', 'spanish').
            voice_id (int): Voice ID 1–10.
        """
        if "speed" in kwargs:
            self.speed = float(kwargs["speed"])
        if "language" in kwargs:
            self.language = kwargs["language"]
        if "voice_id" in kwargs:
            self.voice_id = int(kwargs["voice_id"])
