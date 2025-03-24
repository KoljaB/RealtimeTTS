from .base_engine import BaseEngine
from openai import OpenAI
from typing import Union
import pyaudio
import time

# ANSI escape codes for colors
COLOR_GREEN = "\033[92m"   # Green for text messages
COLOR_YELLOW = "\033[93m"  # Yellow for time messages
COLOR_WHITE = "\033[97m"   # White for all else
COLOR_RESET = "\033[0m"

class OpenAIVoice:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"{self.name}"

class OpenAIEngine(BaseEngine):
    def __init__(
            self,
            model: str = "tts-1",
            voice: str = "nova",
            instructions: str = None,
            debug: bool = False,
            speed: float = None,
            response_format: str = "mp3",
            timeout: float = None
        ):
        """
        Initializes an OpenAI realtime text to speech engine object.

        Args:
            model (str, optional): The model to use for speech synthesis.
              Available models: gpt-4o-mini-tts, tts-1-hd and tts-1. Defaults to "tts-1".
            voice (str, optional): The voice to use for speech synthesis.
              Available voices: "alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"
              Defaults to "nova".
            instructions (str, optional): Instructions to guide speech synthesis.
            debug (bool, optional): If True, prints debugging information.
            speed (float, optional): Optional speech speed for synthesis.
            response_format (str, optional): Audio format for the response. Only "mp3" and "pcm" are allowed.
            timeout (float, optional): Timeout for the API call in seconds.
        """
        self.voices = ["alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"]
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.debug = debug
        self.speed = speed

        # Validate response_format
        if response_format.lower() not in ["mp3", "pcm"]:
            raise ValueError("response_format must be either 'mp3' or 'pcm'")
        self.response_format = response_format.lower()

        self.timeout = timeout

        self.client = OpenAI()
        # Assuming queue is defined elsewhere or should be initialized
        self.queue = None

    def set_instructions(self, instructions: str):
        """
        Sets the instructions to be used for speech synthesis.

        Args:
            instructions (str): The instructions to be used for speech synthesis.
        """
        self.instructions = instructions

    def post_init(self):
        self.engine_name = "openai"

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration information suitable
        for OpenAI Engine.

        Returns:
            tuple: A tuple containing the audio format, number of channels,
              and the sample rate.
              For "pcm" response_format, returns (pyaudio.paInt16, 1, 22050).
              For "mp3", returns (pyaudio.paCustomFormat, 1, 22050).
        """
        if self.response_format == "pcm":
            return pyaudio.paInt16, 1, 22050
        else:
            return pyaudio.paCustomFormat, 1, 22050

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """
        if self.debug:
            print(f"Synthesizing text: \"{COLOR_GREEN}{text}{COLOR_RESET}\"")

        start_time = time.time()
        first_token_printed = False

        # Build parameters for the API call
        params = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "instructions": self.instructions,
            "response_format": self.response_format
        }
        if self.speed is not None:
            params["speed"] = self.speed
        if self.timeout is not None:
            params["timeout"] = self.timeout

        response = self.client.audio.speech.create(**params)

        for data in response.iter_bytes():
            if not first_token_printed:
                elapsed = time.time() - start_time
                if self.debug:
                    print(f"Time to first audio token: {COLOR_YELLOW}{elapsed:.2f}{COLOR_RESET} seconds.")
                first_token_printed = True
            # Write the raw audio data into the queue for the stream player to handle
            self.queue.put(data)

        return True

    def get_voices(self):
        """
        Retrieves the installed voices available for the OpenAI TTS engine.
        """
        voice_objects = []
        for voice in self.voices:
            voice_objects.append(OpenAIVoice(voice))
        return voice_objects

    def set_voice(self, voice: Union[str, object]):
        """
        Sets the voice to be used for speech synthesis.

        Args:
            voice (Union[str, OpenAIVoice]): The voice to be used for speech synthesis.
        """
        if isinstance(voice, OpenAIVoice):
            self.voice = voice.name
        else:
            installed_voices = self.get_voices()
            for installed_voice in installed_voices:
                if voice in installed_voice.name:
                    self.voice = installed_voice.name

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets the voice parameters to be used for speech synthesis.

        Args:
            **voice_parameters: The voice parameters to be used for speech synthesis.
        """
        pass
