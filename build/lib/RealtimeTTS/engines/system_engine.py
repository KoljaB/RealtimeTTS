from .base_engine import BaseEngine
from pydub.utils import mediainfo
from pydub import AudioSegment
from typing import Union
import tempfile
import pyaudio
import pyttsx3
import wave
import os

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
        Initializes a system realtime text to speech engine object.

        Args:
            voice (str, optional): Voice name. Defaults to "Zira".
            print_installed_voices (bool, optional): Indicates if the list of installed voices should be printed. Defaults to False.
        """        

        self.engine = pyttsx3.init()
        self.set_voice(voice)
        temp_file_path = tempfile.gettempdir()
        self.file_path = os.path.join(temp_file_path, SYNTHESIS_FILE)

        if print_installed_voices:
            print (self.get_voices())

    def post_init(self):
        self.engine_name = "system"

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration information suitable for System Engine.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
                  - Format (int): The format of the audio stream. pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels. 1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz. 16000 represents 16kHz sample rate.
        """        
        return pyaudio.paInt16, 1, 22050
    
    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """

        self.engine.save_to_file(text, self.file_path)
        self.engine.runAndWait()

        # Get media info of the file
        info = mediainfo(self.file_path)

        # Check if the file format is AIFF and convert to WAV if necessary
        if info['format_name'] == 'aiff':
            audio = AudioSegment.from_file(self.file_path, format="aiff")
            audio.export(self.file_path, format="wav")

        # Now open the WAV file
        with wave.open(self.file_path, 'rb') as wf:
            audio_data = wf.readframes(wf.getnframes())
            self.queue.put(audio_data)
            return True

        # Return False if the process failed
        return False

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

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets the voice parameters to be used for speech synthesis.

        Args:
            **voice_parameters: The voice parameters to be used for speech synthesis.
        """
        for parameter, value in voice_parameters.items():
            self.engine.setProperty(parameter, value)