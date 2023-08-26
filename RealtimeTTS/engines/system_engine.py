from .base_engine import BaseEngine
import pyaudio
import pyttsx3
import os
import tempfile
import wave
from typing import Union

SYNTHESIS_FILE = 'system_speech_synthesis.wav' 


class SystemVoice:
    def __init__(self, name, id):
        self.name = name
        self.id = id

    def __repr__(self):
        return self.name
        #return f"<Voice(name={self.name})>"


class SystemEngine(BaseEngine):

    def __init__(self, 
                 voice: str = "Zira",
                 print_installed_voices: bool = False):
        """
        Initializes a system voice realtime text to speech engine object.

        Args:
            speech_key (str): Azure subscription key. (TTS API key)
            service_region (str): Azure service region. (Cloud Region ID)
            voice (str, optional): Voice name. Defaults to "en-US-AriaNeural".
            rate (float, optional): Speech speed. Defaults to "0.0".
            pitch (float, optional): Speech pitch. Defaults to "0.0".
        """        

        self.engine = pyttsx3.init()
        self.set_voice(voice)
        temp_file_path = tempfile.gettempdir()
        self.file_path = os.path.join(temp_file_path, SYNTHESIS_FILE)

    def get_stream_info(self):
        """
        Returns the audio stream configuration information suitable for PyAudio.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
                  - Format (int): The format of the audio stream. pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels. 1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz. 16000 represents 16kHz sample rate.
        """        
        return pyaudio.paInt16, 1, 22050

    def synthesize(self, 
                   text: str):
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """

        self.engine.save_to_file(text, self.file_path)
        self.engine.runAndWait()

        # Open the saved WAV file
        with wave.open(self.file_path, 'rb') as wf:
            audio_data = wf.readframes(wf.getnframes())
            self.queue.put(audio_data)           

    def get_voices(self):
        """
        Retrieves the voices available in the underlying system's speech engine.

        This method fetches the list of available voices from the system's speech engine 
        (represented by the `engine` attribute of the instance) and wraps each voice's information 
        in a `SystemVoice` object.

        Returns:
            list[SystemVoice]: A list containing SystemVoice objects representing each available voice. 
                            Each SystemVoice object encapsulates information like the name and ID of the voice.

        Note:
            Ensure that the `engine` attribute of the instance is properly initialized before calling this method.
        """        
        voice_objects = []
        voices = self.engine.getProperty('voices')
        for voice in voices:
            voice_object = SystemVoice(voice.name, voice.id)
            voice_objects.append(voice_object)
        return voice_objects
    
    def set_voice(self, voice: Union[str, SystemVoice]):
        """
        Sets the voice to be used for speech synthesis.

        Args:
            voice (Union[str, SystemVoice]): The voice to be used for speech synthesis.
        """
        if isinstance(voice, SystemVoice):
            self.engine.setProperty('voice', voice.id)            
        else:
            installed_voices = self.engine.getProperty('voices')
            if not voice is None:
                for installed_voice in installed_voices:
                    if voice in installed_voice.name:
                        self.engine.setProperty('voice', installed_voice.id)