from elevenlabs.client import ElevenLabs
from typing import Union
from .base_engine import BaseEngine
import logging
import pyaudio
import os
import traceback


class ElevenlabsVoice:
    def __init__(self, name, voice_id, category, description, labels):
        self.name = name
        self.voice_id = voice_id
        self.category = category
        self.description = description
        self.labels = labels

    def __repr__(self):
        label_string = ", ".join(self.labels.values())
        return f"{self.name} ({self.category}, id: {self.voice_id}, {label_string})"


class ElevenlabsEngine(BaseEngine):
    def __init__(
        self,
        api_key: str = "",
        voice: str = "Nicole",
        id: str = "piTKgcLEGmPE4e6mEKli",
        category: str = "premade",
        clarity: float = 75.0,
        stability: float = 50.0,
        style_exxageration: float = 0.0,
        model: str = "eleven_multilingual_v2",
    ):
        """
        Initializes an elevenlabs voice realtime text to speech engine object.

        Args:
            api_key (str): Elevenlabs API key. (TTS API key)
            voice (str, optional): Voice name. Defaults to "Nicole".
            id (str, optional): Voice ID. Defaults to "piTKgcLEGmPE4e6mEKli".
            category (str, optional): Voice category. Defaults to "premade".
            clarity (float, optional): Clarity / Similarity. Adjusts voice similarity and resemblance. Defaults to "75.0".
            stability (float, optional): Stability. Controls the voice performance, with higher values producing a steadier tone and lower values giving a more emotive output. Defaults to "50.0".
            style_exxageration (float, optional): Style Exxageration. Controls the voice performance, with higher values giving a more emotive output and lower values producing a steadier tone. Defaults to "0.0".
            model (str, optional): Model. Defaults to "eleven_multilingual_v2". Some models may not work with real time inference.
        """

        self.voice_name = voice
        self.id = id
        self.category = category
        self.clarity = clarity
        self.stability = stability
        self.style_exxageration = style_exxageration
        self.model = model
        if not api_key:
            api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing ElevenLabs API key. Either:\n"
                "1. Pass the API key as a parameter\n" 
                "2. Set ELEVENLABS_API_KEY environment variable"
            )

        self.client = ElevenLabs(api_key=api_key)

    def post_init(self):
        """Set engine name and generator consumption capability."""
        self.can_consume_generators = False
        self.engine_name = "elevenlabs"

    def get_stream_info(self):
        """
        Returns the audio stream configuration information suitable for PyAudio.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
                  - Format (int): The format of the audio stream. pyaudio.paCustomFormat for mpeg.
                  - Channels (int): The number of audio channels. -1 for mpeg.
                  - Sample Rate (int): The sample rate of the audio in Hz. -1 for mpeg.
        """
        return pyaudio.paCustomFormat, -1, -1

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """
        super().synthesize(text)

        # NOTE: The new elevenlabs API (v1.0.0+) does not allow setting
        # voice settings (stability, clarity, etc.) per-request on the stream endpoint.
        # These settings are now configured with the voice itself in the Voice Lab
        # or via the API for voice editing. The parameters in __init__ are kept
        # for compatibility but are not used in this method.
        
        try:
            audio_stream = self.client.text_to_speech.stream(
                text=text,
                voice_id=self.id,
                model_id=self.model,
            )

            for chunk in audio_stream:
                if chunk:
                    self.queue.put(chunk)
            return True
        
        except Exception as e:
            logging.error(f"Elevenlabs synthesis error: {e}")
            traceback.print_exc()
            return False

    def set_api_key(self, api_key: str):
        """
        Sets the elevenlabs api key.

        Args:
            api_key (str): Elevenlabs API key. (TTS API key)
        """
        self.api_key = api_key
        if api_key:
            self.client = ElevenLabs(api_key=api_key)

    def get_voices(self):
        """
        Retrieves the voices available from the Elevenlabs voice source.

        Calling this takes time, it sends a request to the Elevenlabs API to fetch the list of available voices.
        This method fetches the list of available voices using the elevenlabs `voices()` function and then
        constructs a list of `ElevenlabsVoice` objects to represent each voice's details.

        Returns:
            list[ElevenlabsVoice]: A list containing ElevenlabsVoice objects representing each available voice.
                                Each ElevenlabsVoice object encapsulates information such as the voice's name,
                                ID, category, description, and associated labels.

        Note:
            This method relies on the `voices()` function to obtain the raw voice data. Ensure that the
            `voices()` function is accessible and functional before calling this method.
        """
        fetched_voices = self.client.voices.search()

        voices_list = fetched_voices.voices

        voice_objects = []
        for voice in voices_list:
            voice_object = ElevenlabsVoice(
                name=voice.name,
                voice_id=voice.voice_id,
                category=voice.category,
                description=voice.description,
                labels=voice.labels,
            )
            voice_objects.append(voice_object)

        return voice_objects

    def set_voice(self, voice: Union[str, "ElevenlabsVoice"]):
        """
        Sets the voice to be used for speech synthesis.

        Args:
            voice (Union[str, ElevenlabsVoice]): The voice to be used for speech synthesis.
        """
        if isinstance(voice, ElevenlabsVoice):
            logging.info(f"Setting voice to {voice.name}")
            self.voice_name = voice.name
            self.id = voice.voice_id
            self.category = voice.category
            return
        else:
            installed_voices = self.get_voices()
            for installed_voice in installed_voices:
                if voice in installed_voice.name:
                    logging.info(f"Setting voice to {installed_voice.name}")
                    self.voice_name = installed_voice.name
                    self.id = installed_voice.voice_id
                    self.category = installed_voice.category
                    return

        logging.warning(f"Voice {voice} not found.")

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets the voice parameters to be used for speech synthesis.

        Args:
            **voice_parameters: The voice parameters to be used for speech synthesis.
        """
        if "clarity" in voice_parameters:
            self.clarity = voice_parameters["clarity"]
        if "stability" in voice_parameters:
            self.stability = voice_parameters["stability"]
        if "style_exxageration" in voice_parameters:
            self.style_exxageration = voice_parameters["style_exxageration"]
            
# from elevenlabs.client import ElevenLabs
# from elevenlabs import Voice, VoiceSettings
# from typing import Iterator, Union
# from .base_engine import BaseEngine
# import subprocess
# import threading
# import logging
# import pyaudio
# import os


# class ElevenlabsVoice:
#     def __init__(self, name, voice_id, category, description, labels):
#         self.name = name
#         self.voice_id = voice_id
#         self.category = category
#         self.description = description
#         self.labels = labels

#     def __repr__(self):
#         label_string = ", ".join(self.labels.values())
#         return f"{self.name} ({self.category}, id: {self.voice_id}, {label_string})"


# class ElevenlabsEngine(BaseEngine):
#     def __init__(
#         self,
#         api_key: str = "",
#         voice: str = "Nicole",
#         id: str = "piTKgcLEGmPE4e6mEKli",
#         category: str = "premade",
#         clarity: float = 75.0,
#         stability: float = 50.0,
#         style_exxageration: float = 0.0,
#         model: str = "eleven_multilingual_v1",
#     ):
#         """
#         Initializes an elevenlabs voice realtime text to speech engine object.

#         Args:
#             api_key (str): Elevenlabs API key. (TTS API key)
#             voice (str, optional): Voice name. Defaults to "Nicole".
#             id (str, optional): Voice ID. Defaults to "piTKgcLEGmPE4e6mEKli".
#             category (str, optional): Voice category. Defaults to "premade".
#             clarity (float, optional): Clarity / Similarity. Adjusts voice similarity and resemblance. Defaults to "75.0".
#             stability (float, optional): Stability. Controls the voice performance, with higher values producing a steadier tone and lower values giving a more emotive output. Defaults to "50.0".
#             style_exxageration (float, optional): Style Exxageration. Controls the voice performance, with higher values giving a more emotive output and lower values producing a steadier tone. Defaults to "0.0".
#             model (str, optional): Model. Defaults to "eleven_multilingual_v1". Some models may not work with real time inference.
#         """

#         self.voice_name = voice
#         self.id = id
#         self.category = category
#         self.clarity = clarity
#         self.stability = stability
#         self.style_exxageration = style_exxageration
#         self.model = model
#         if not api_key:
#             api_key = os.environ.get("ELEVENLABS_API_KEY")
#         if not api_key:
#             raise ValueError(
#                 "Missing ElevenLabs API key. Either:\n"
#                 "1. Pass the API key as a parameter\n" 
#                 "2. Set ELEVENLABS_API_KEY environment variable"
#             )

#         self.client = ElevenLabs(api_key=api_key)

#     def post_init(self):
#         """Information that this engine can handle generators directly"""
#         self.can_consume_generators = True
#         self.engine_name = "elevenlabs"

#     def get_stream_info(self):
#         """
#         Returns the audio stream configuration information suitable for PyAudio.

#         Returns:
#             tuple: A tuple containing the audio format, number of channels, and the sample rate.
#                   - Format (int): The format of the audio stream. pyaudio.paInt16 represents 16-bit integers.
#                   - Channels (int): The number of audio channels. 1 represents mono audio.
#                   - Sample Rate (int): The sample rate of the audio in Hz. 16000 represents 16kHz sample rate.
#         """
#         return pyaudio.paCustomFormat, -1, -1

#     def synthesize(self, generator: Iterator[str]) -> bool:
#         """
#         Synthesizes text to audio stream.

#         Args:
#             text (str): Text to synthesize.
#         """
#         voice = Voice(
#             voice_id=self.id,
#             settings=VoiceSettings(
#                 stability=self.stability / 100,
#                 similarity_boost=self.clarity / 100,
#                 style=self.style_exxageration / 100,
#                 use_speaker_boost=True,
#             ),
#         )

#         self.audio_stream = self.client.generate(
#             text=generator, voice=voice, model=self.model, stream=True
#         )

#         # self.stream(self.audio_stream)
#         for chunk in self.audio_stream:
#             if chunk is not None:
#                 self.queue.put(chunk)

#         return True

#     def set_api_key(self, api_key: str):
#         """
#         Sets the elevenlabs api key.

#         Args:
#             api_key (str): Elevenlabs API key. (TTS API key)
#         """
#         self.api_key = api_key
#         if api_key:
#             self.client = ElevenLabs(api_key=api_key)

#     def get_voices(self):
#         """
#         Retrieves the voices available from the Elevenlabs voice source.

#         Calling this takes time, it sends a request to the Elevenlabs API to fetch the list of available voices.
#         This method fetches the list of available voices using the elevenlabs `voices()` function and then
#         constructs a list of `ElevenlabsVoice` objects to represent each voice's details.

#         Returns:
#             list[ElevenlabsVoice]: A list containing ElevenlabsVoice objects representing each available voice.
#                                 Each ElevenlabsVoice object encapsulates information such as the voice's name,
#                                 ID, category, description, and associated labels.

#         Note:
#             This method relies on the `voices()` function to obtain the raw voice data. Ensure that the
#             `voices()` function is accessible and functional before calling this method.
#         """
#         fetched_voices = self.client.voices.get_all()

#         voices_list = fetched_voices.voices

#         voice_objects = []
#         for voice in voices_list:
#             voice_object = ElevenlabsVoice(
#                 name=voice.name,
#                 voice_id=voice.voice_id,
#                 category=voice.category,
#                 description=voice.description,
#                 labels=voice.labels,
#             )
#             voice_objects.append(voice_object)

#         return voice_objects

#     def set_voice(self, voice: Union[str, ElevenlabsVoice]):
#         """
#         Sets the voice to be used for speech synthesis.

#         Args:
#             voice (Union[str, ElevenlabsVoice]): The voice to be used for speech synthesis.
#         """
#         if isinstance(voice, ElevenlabsVoice):
#             logging.info(f"Setting voice to {voice.name}")
#             self.voice_name = voice.name
#             self.id = voice.voice_id
#             self.category = voice.category
#             return
#         else:
#             installed_voices = self.get_voices()
#             for installed_voice in installed_voices:
#                 if voice in installed_voice.name:
#                     logging.info(f"Setting voice to {installed_voice.name}")
#                     self.voice_name = installed_voice.name
#                     self.id = installed_voice.voice_id
#                     self.category = installed_voice.category
#                     return

#         logging.warning(f"Voice {voice} not found.")

#     def set_voice_parameters(self, **voice_parameters):
#         """
#         Sets the voice parameters to be used for speech synthesis.

#         Args:
#             **voice_parameters: The voice parameters to be used for speech synthesis.
#         """
#         if "clarity" in voice_parameters:
#             self.clarity = voice_parameters["clarity"]
#         if "stability" in voice_parameters:
#             self.stability = voice_parameters["stability"]
#         if "style_exxageration" in voice_parameters:
#             self.style_exxageration = voice_parameters["style_exxageration"]
