from .base_engine import BaseEngine
from openai import OpenAI
from typing import Union
import pyaudio

class OpenAIEngine(BaseEngine):

    def __init__(self, 
                 model: str = "tts-1",
                 voice: str = "nova"):
        """
        Initializes a openai realtime text to speech engine object.

        Args:
            model (str, optional): The model to use for speech synthesis. Available models: tts-1-hd and tts-1. Defaults to "tts-1".
            voice (str, optional): The voice to use for speech synthesis. Available voices: alloy, echo, fable, onyx, nova, and shimmer. Defaults to "alloy".
        """

        self.model = model
        self.voice = voice
        self.client = OpenAI()

    def post_init(self):
        self.engine_name = "openai"

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration information suitable for OpenAI Engine.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
                  - Format (int): The format of the audio stream. pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels. 1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz. 16000 represents 16kHz sample rate.
        """        
        return pyaudio.paCustomFormat, 1, 22050

    def synthesize(self, 
                   text: str) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """
        response = self.client.audio.speech.create(
            model = self.model,
            voice = self.voice,
            input = text,
        )

        for data in response.iter_bytes():
            # we write the mp3 raw data into the queue and let stream player handle the rest
            self.queue.put(data)

        return True

    def get_voices(self):
        """
        Retrieves the installed voices available for the Coqui TTS engine.
        """
        # since we currently can't retrieve the voices from the engine api, we return an empty list
        # we do NOT track the voices manually, since these are the types of maintenance tasks that add up long term
        voice_objects = []
        return voice_objects
    
    def set_voice(self, voice: Union[str, object]):
        """
        Sets the voice to be used for speech synthesis.

        Args:
            voice (Union[str, SystemVoice]): The voice to be used for speech synthesis.
        """
        pass

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets the voice parameters to be used for speech synthesis.

        Args:
            **voice_parameters: The voice parameters to be used for speech synthesis.
        """
        pass