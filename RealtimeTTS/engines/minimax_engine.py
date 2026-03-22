from .base_engine import BaseEngine
from typing import Union
import requests
import pyaudio
import time
import os
import logging

# ANSI escape codes for colors
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_WHITE = "\033[97m"
COLOR_RESET = "\033[0m"


class MiniMaxVoice:
    def __init__(self, name, language="english"):
        self.name = name
        self.language = language

    def __repr__(self):
        return f"{self.name}"


class MiniMaxEngine(BaseEngine):
    """
    MiniMax Cloud text-to-speech engine for RealtimeTTS.

    Uses the MiniMax T2A v2 API to convert text to speech with support for
    multiple voice presets and two model variants (speech-2.8-hd for quality,
    speech-2.8-turbo for speed).

    API Docs: https://platform.minimaxi.com/document/T2A%20V2
    """

    # Available voice presets
    VOICES = [
        {"name": "English_Graceful_Lady", "language": "english"},
        {"name": "English_Insightful_Speaker", "language": "english"},
        {"name": "English_radiant_girl", "language": "english"},
        {"name": "English_Persuasive_Man", "language": "english"},
        {"name": "English_Lucky_Robot", "language": "english"},
        {"name": "Wise_Woman", "language": "multilingual"},
        {"name": "cute_boy", "language": "multilingual"},
        {"name": "lovely_girl", "language": "multilingual"},
        {"name": "Friendly_Person", "language": "multilingual"},
        {"name": "Inspirational_girl", "language": "multilingual"},
        {"name": "Deep_Voice_Man", "language": "multilingual"},
        {"name": "sweet_girl", "language": "multilingual"},
    ]

    def __init__(
        self,
        api_key: str = None,
        model: str = "speech-2.8-hd",
        voice: str = "English_Graceful_Lady",
        speed: float = 1.0,
        volume: float = 1.0,
        pitch: int = 0,
        debug: bool = False,
    ):
        """
        Initializes a MiniMax Cloud realtime text to speech engine object.

        Args:
            api_key (str, optional): MiniMax API key. Defaults to MINIMAX_API_KEY env var.
            model (str, optional): TTS model to use.
                Available models: "speech-2.8-hd" (high quality), "speech-2.8-turbo" (fast).
                Defaults to "speech-2.8-hd".
            voice (str, optional): Voice preset to use. Defaults to "English_Graceful_Lady".
                See VOICES list for available presets.
            speed (float, optional): Speech speed (0.5 to 2.0). Defaults to 1.0.
            volume (float, optional): Audio volume (0.1 to 10.0). Defaults to 1.0.
            pitch (int, optional): Pitch adjustment (-12 to 12 semitones). Defaults to 0.
            debug (bool, optional): If True, prints debugging information.
        """
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MiniMax API key is required. Provide it via api_key parameter "
                "or MINIMAX_API_KEY environment variable."
            )

        self.model = model
        self.voice = voice
        self.speed = speed
        self.volume = volume
        self.pitch = pitch
        self.debug = debug
        self.queue = None

        self.base_url = "https://api.minimax.io/v1/t2a_v2"

    def post_init(self):
        self.engine_name = "minimax"

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration for MiniMax audio.
        MiniMax returns MP3 audio.

        Returns:
            tuple: (format, channels, sample_rate)
        """
        return pyaudio.paCustomFormat, 1, 32000

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.

        Returns:
            bool: True if synthesis succeeded.
        """
        super().synthesize(text)

        if self.debug:
            print(f"MiniMax synthesizing: \"{COLOR_GREEN}{text}{COLOR_RESET}\"")

        start_time = time.time()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "text": text,
            "voice_setting": {
                "voice_id": self.voice,
                "speed": self.speed,
                "vol": self.volume,
                "pitch": self.pitch,
            },
            "audio_setting": {
                "format": "mp3",
            },
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()

            # Check for API errors
            base_resp = result.get("base_resp", {})
            status_code = base_resp.get("status_code", 0)
            if status_code != 0:
                error_msg = base_resp.get("status_msg", "Unknown error")
                logging.error(f"MiniMax TTS API error: {error_msg} (code: {status_code})")
                if self.debug:
                    print(f"{COLOR_YELLOW}MiniMax API error: {error_msg}{COLOR_RESET}")
                return False

            # Extract audio data (hex-encoded MP3 bytes)
            audio_hex = result.get("data", {}).get("audio", "")
            if not audio_hex:
                logging.error("MiniMax TTS returned empty audio data")
                if self.debug:
                    print(f"{COLOR_YELLOW}MiniMax returned empty audio{COLOR_RESET}")
                return False

            # Decode hex string to bytes
            audio_bytes = bytes.fromhex(audio_hex)

            elapsed = time.time() - start_time
            if self.debug:
                print(
                    f"MiniMax time to audio: {COLOR_YELLOW}{elapsed:.2f}{COLOR_RESET} seconds "
                    f"({len(audio_bytes)} bytes)"
                )

            if self.stop_synthesis_event.is_set():
                return False

            # Put audio data in queue (MP3 bytes)
            self.queue.put(audio_bytes)

            return True

        except requests.exceptions.RequestException as e:
            logging.error(f"MiniMax TTS request error: {e}")
            if self.debug:
                print(f"{COLOR_YELLOW}MiniMax request error: {e}{COLOR_RESET}")
            return False
        except ValueError as e:
            logging.error(f"MiniMax TTS decode error: {e}")
            if self.debug:
                print(f"{COLOR_YELLOW}MiniMax decode error: {e}{COLOR_RESET}")
            return False

    def get_voices(self):
        """
        Retrieves the available voices for MiniMax TTS.

        Returns:
            list: List of MiniMaxVoice objects.
        """
        voice_objects = []
        for voice_data in self.VOICES:
            voice_objects.append(
                MiniMaxVoice(
                    name=voice_data["name"],
                    language=voice_data["language"],
                )
            )
        return voice_objects

    def set_voice(self, voice: Union[str, "MiniMaxVoice"]):
        """
        Sets the voice to be used for speech synthesis.

        Args:
            voice (Union[str, MiniMaxVoice]): The voice to use.
        """
        if isinstance(voice, MiniMaxVoice):
            self.voice = voice.name
        else:
            # Check if the voice name matches any known voice
            for v in self.VOICES:
                if voice.lower() == v["name"].lower():
                    self.voice = v["name"]
                    return
            # Allow arbitrary voice IDs
            self.voice = voice

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets voice parameters for synthesis.

        Args:
            **voice_parameters: Parameters like speed, volume, pitch, model.
        """
        if "speed" in voice_parameters:
            self.speed = voice_parameters["speed"]
        if "volume" in voice_parameters:
            self.volume = voice_parameters["volume"]
        if "pitch" in voice_parameters:
            self.pitch = voice_parameters["pitch"]
        if "model" in voice_parameters:
            self.model = voice_parameters["model"]

    def shutdown(self):
        """
        Shuts down the engine and releases resources.
        """
        pass
