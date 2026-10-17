import logging
import os
import time
from typing import ClassVar, Union

import pyaudio
import requests

from .base_engine import BaseEngine

logger = logging.getLogger(__name__)


class AtlasCloudVoice:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class AtlasCloudEngine(BaseEngine):
    """Text-to-speech engine backed by Atlas Cloud's async audio API."""

    VOICES: ClassVar[list[str]] = [
        "English_expressive_narrator",
        "English_radiant_girl",
        "English_magnetic_voiced_man",
        "English_compelling_lady1",
        "English_Aussie_Bloke",
        "English_captivating_female1",
        "English_Upbeat_Woman",
        "English_Trustworth_Man",
        "English_CalmWoman",
        "English_UpsetGirl",
        "English_Gentle-voiced_man",
        "English_Whispering_girl",
        "English_Diligent_Man",
        "English_Graceful_Lady",
        "English_ReservedYoungMan",
    ]

    def __init__(
        self,
        api_key=None,
        model="minimax/speech-2.6-turbo",
        voice="English_expressive_narrator",
        speed=1.0,
        volume=1.0,
        pitch=0,
        sample_rate=32000,
        poll_interval=0.5,
        timeout=60,
        debug=False,
    ):
        self.api_key = api_key or os.environ.get("ATLASCLOUD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Atlas Cloud API key is required. Provide api_key or set "
                "ATLASCLOUD_API_KEY."
            )
        self.model = model
        self.voice = voice
        self.speed = speed
        self.volume = volume
        self.pitch = pitch
        self.sample_rate = sample_rate
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.debug = debug
        self.base_url = "https://api.atlascloud.ai"

    def post_init(self):
        self.engine_name = "atlascloud"

    def get_stream_info(self):
        return pyaudio.paCustomFormat, 1, self.sample_rate

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)
        payload = {
            "model": self.model,
            "text": text,
            "voice": self.voice,
            "speed": self.speed,
            "vol": self.volume,
            "pitch": self.pitch,
            "format": "mp3",
            "sample_rate": self.sample_rate,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/v1/model/generateAudio",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            response_body = response.json()
            prediction = response_body.get("data", response_body)
            prediction_id = prediction.get("id")
            if not prediction_id:
                raise ValueError("Atlas Cloud response did not include a prediction id")

            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if self.stop_synthesis_event.is_set():
                    return False
                result_response = requests.get(
                    f"{self.base_url}/api/v1/model/prediction/{prediction_id}",
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                result_response.raise_for_status()
                result_body = result_response.json()
                result = result_body.get("data", result_body)
                status = result.get("status", "").lower()
                if status in ("completed", "succeeded"):
                    outputs = result.get("outputs") or []
                    if not outputs:
                        raise ValueError("Atlas Cloud prediction had no audio output")
                    audio_response = requests.get(
                        outputs[0], stream=True, timeout=self.timeout
                    )
                    audio_response.raise_for_status()
                    for chunk in audio_response.iter_content(chunk_size=8192):
                        if self.stop_synthesis_event.is_set():
                            return False
                        if chunk:
                            self.queue.put(chunk)
                    return True
                if status == "failed":
                    logger.error("Atlas Cloud TTS prediction failed: %s", result)
                    return False
                time.sleep(self.poll_interval)

            logger.error("Atlas Cloud TTS prediction timed out")
            return False
        except (requests.RequestException, ValueError) as exc:
            logger.error("Atlas Cloud TTS request failed: %s", exc)
            return False

    def get_voices(self):
        return [AtlasCloudVoice(name) for name in self.VOICES]

    def set_voice(self, voice: Union[str, AtlasCloudVoice]):  # noqa: UP007
        self.voice = voice.name if isinstance(voice, AtlasCloudVoice) else voice

    def set_voice_parameters(self, **voice_parameters):
        for source, target in (
            ("speed", "speed"),
            ("volume", "volume"),
            ("pitch", "pitch"),
            ("model", "model"),
        ):
            if source in voice_parameters:
                setattr(self, target, voice_parameters[source])

    def shutdown(self):
        pass
