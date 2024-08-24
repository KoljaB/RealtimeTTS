from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings
from typing import Iterator, Union
from .base_engine import BaseEngine
import subprocess
import threading
import logging
import pyaudio

class ElevenlabsVoice:
    def __init__(self, name, voice_id, category, description, labels):
        self.name = name
        self.voice_id = voice_id
        self.category = category
        self.description = description
        self.labels = labels

    def __repr__(self):
        label_string = ', '.join(self.labels.values())
        return f"{self.name} ({self.category}, id: {self.voice_id}, {label_string})"


class ElevenlabsEngine(BaseEngine):

    def __init__(self, 
                 api_key: str = "", 
                 voice: str = "Nicole", 
                 id: str = "piTKgcLEGmPE4e6mEKli",
                 category: str = "premade",
                 clarity: float = 75.0,
                 stability: float = 50.0,
                 style_exxageration: float = 0.0,
                 model: str = "eleven_multilingual_v1"):
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
            model (str, optional): Model. Defaults to "eleven_multilingual_v1". Some models may not work with real time inference.
        """

        self.voice_name = voice
        self.id = id
        self.category = category
        self.clarity = clarity
        self.stability = stability
        self.style_exxageration = style_exxageration
        self.model = model
        self.pause_event = threading.Event()
        self.immediate_stop = threading.Event()
        self.on_audio_chunk = None
        self.muted = False
        self.on_playback_started = False

        self.client = ElevenLabs(
            api_key=api_key
        )

    def post_init(self):
        """ Information that this engine can handle generators directly """ 
        self.can_consume_generators = True
        self.engine_name = "elevenlabs"

    def get_stream_info(self):
        """
        Returns the audio stream configuration information suitable for PyAudio.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
                  - Format (int): The format of the audio stream. pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels. 1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz. 16000 represents 16kHz sample rate.
        """        
        return pyaudio.paCustomFormat, -1, -1
    
    def pause(self):
        """Pauses playback of the synthesized audio stream (won't work properly with elevenlabs)."""
        self.pause_event.set()

    def set_muted(self, muted: bool):
        """Mutes the audio stream."""
        self.muted = muted

    def resume(self):
        """Resumes a previously paused playback of the synthesized audio stream (won't work properly with elevenlabs)."""
        self.pause_event.clear()

    def stop(self):
        """Stops the playback of the synthesized audio stream immediately."""
        self.mpv_process.terminate()
        return True

    def synthesize(self, 
                   generator: Iterator[str]) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """
        
        self.on_playback_started = False
        self.immediate_stop.clear()

        voice = Voice(
            voice_id=self.id,
            settings=VoiceSettings(
                stability=self.stability / 100,
                similarity_boost=self.clarity / 100,
                style=self.style_exxageration / 100,
                use_speaker_boost=True
            )
        )

        self.audio_stream = self.client.generate(
            text=generator,
            voice=voice,
            model=self.model,
            stream=True
        )

        self.stream(self.audio_stream)

        return True
        
    def set_api_key(self, api_key: str):
        """
        Sets the elevenlabs api key. 

        Args:
            api_key (str): Elevenlabs API key. (TTS API key)
        """
        self.api_key = api_key
        if api_key:
            self.client = ElevenLabs(
               api_key=api_key
            )

    def stream(self, audio_stream: Iterator[bytes]) -> bytes:
        """
        Stream the audio data using the 'mpv' player.

        This method takes the audio_stream iterator, which contains bytes of audio data,
        and plays them using the 'mpv' player. The function will continuously feed the
        'mpv' process with chunks of audio data until the audio is finished or an error
        occurs.

        If the 'mpv' player is not installed, a ValueError is raised.

        Args:
            audio_stream (Iterator[bytes]): An iterator that yields bytes of audio data.
            filename (Optional[str], optional): The filename to save the audio to. Defaults to None.

        Returns:
            bytes: The concatenated bytes of the entire audio stream.

        Raises:
            ValueError: If the 'mpv' player is not found.
        """       
        if not self.is_installed("mpv"):
            message = (
                "mpv not found, necessary to stream audio. "
                "On mac you can install it with 'brew install mpv'. "
                "On linux and windows you can install it from https://mpv.io/"
            )
            raise ValueError(message)

        if not self.muted:
            mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
            self.mpv_process = subprocess.Popen(
                mpv_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        audio = b""

        try: 
            for chunk in audio_stream:
                if chunk is not None:
                    if not self.on_playback_started and self.on_playback_start:
                        self.on_playback_start()
                    self.on_playback_started = True
                    if not self.muted:
                        self.mpv_process.stdin.write(chunk)
                        self.mpv_process.stdin.flush()
                    audio += chunk
                    if self.on_audio_chunk: 
                        self.on_audio_chunk(chunk)
        except BrokenPipeError:
            pass

        if not self.muted:
            if self.mpv_process.stdin:
                self.mpv_process.stdin.close()
            self.mpv_process.wait()

        return audio            
    
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
        fetched_voices = self.client.voices.get_all()

        voices_list = fetched_voices.voices

        voice_objects = []
        for voice in voices_list:
            voice_object = ElevenlabsVoice(
                name=voice.name,
                voice_id=voice.voice_id,
                category=voice.category,
                description=voice.description,
                labels=voice.labels
            )
            voice_objects.append(voice_object)

        return voice_objects

    def set_voice(self, voice: Union[str, ElevenlabsVoice]):
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
        if 'clarity' in voice_parameters:
            self.clarity = voice_parameters['clarity']
        if 'stability' in voice_parameters:
            self.stability = voice_parameters['stability']
        if 'style_exxageration' in voice_parameters:
            self.style_exxageration = voice_parameters['style_exxageration']