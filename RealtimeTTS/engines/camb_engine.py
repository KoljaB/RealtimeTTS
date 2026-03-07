from camb.client import CambAI
from camb.types import StreamTtsOutputConfiguration
from typing import Union
from .base_engine import BaseEngine
import logging
import pyaudio
import os
import traceback


class CambVoice:
    def __init__(self, voice_id, name=None):
        self.voice_id = voice_id
        self.name = name or str(voice_id)

    def __repr__(self):
        return f"{self.name} (id: {self.voice_id})"


class CambEngine(BaseEngine):
    def __init__(
        self,
        api_key: str = "",
        voice_id: int = 147320,
        language: str = "en-us",
        speech_model: str = "mars-flash",
        user_instructions: str = None,
        output_format: str = "wav",
    ):
        """
        Initializes a CAMB AI (MARS) realtime text to speech engine object.

        Args:
            api_key (str): CAMB AI API key.
            voice_id (int): Voice ID. Defaults to 147320.
            language (str): BCP-47 language code. Defaults to "en-us".
            speech_model (str): Model name. One of "mars-flash", "mars-pro", "mars-instruct".
            user_instructions (str): Instructions for mars-instruct model only.
            output_format (str): Output format ("wav" or "mp3"). Defaults to "wav".
        """

        self.voice_id = voice_id
        self.language = language
        self.speech_model = speech_model
        self.user_instructions = user_instructions
        self.output_format = output_format

        if not api_key:
            api_key = os.environ.get("CAMB_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing CAMB AI API key. Either:\n"
                "1. Pass the API key as a parameter\n"
                "2. Set CAMB_API_KEY environment variable"
            )

        self.client = CambAI(api_key=api_key)

    def post_init(self):
        self.engine_name = "camb"
        self.can_consume_generators = False

    def get_stream_info(self):
        return pyaudio.paCustomFormat, -1, -1

    def synthesize(self, text: str) -> bool:
        super().synthesize(text)

        try:
            tts_kwargs = {
                "text": text,
                "language": self.language,
                "voice_id": self.voice_id,
                "speech_model": self.speech_model,
                "output_configuration": StreamTtsOutputConfiguration(format=self.output_format),
            }

            if self.user_instructions and self.speech_model == "mars-instruct":
                tts_kwargs["user_instructions"] = self.user_instructions

            audio_stream = self.client.text_to_speech.tts(**tts_kwargs)

            for chunk in audio_stream:
                if chunk:
                    self.queue.put(chunk)
            return True

        except Exception as e:
            logging.error(f"CAMB AI synthesis error: {e}")
            traceback.print_exc()
            return False

    def get_voices(self):
        return []

    def set_voice(self, voice: Union[int, "CambVoice"]):
        if isinstance(voice, CambVoice):
            self.voice_id = voice.voice_id
        elif isinstance(voice, int):
            self.voice_id = voice
        else:
            logging.warning(f"Invalid voice type: {type(voice)}. Expected CambVoice or int.")

    def set_voice_parameters(self, **voice_parameters):
        if "language" in voice_parameters:
            self.language = voice_parameters["language"]
        if "speech_model" in voice_parameters:
            self.speech_model = voice_parameters["speech_model"]
        if "user_instructions" in voice_parameters:
            self.user_instructions = voice_parameters["user_instructions"]
