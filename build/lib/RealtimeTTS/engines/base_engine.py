from abc import ABCMeta, ABC, abstractmethod
from typing import Union
import shutil
import queue

# Define a meta class that will automatically call the BaseEngine's __init__ method
# and also the post_init method if it exists.
class BaseInitMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        # Create an instance of the class that this meta class is used on.
        instance = super().__call__(*args, **kwargs)
        
        # Call the __init__ method of BaseEngine to set default properties.
        BaseEngine.__init__(instance)
                
        # If the instance has a post_init method, call it.
        # This allows subclasses to define additional initialization steps.
        if hasattr(instance, "post_init"):
            instance.post_init()

        return instance

# Define a base class for engines with the custom meta class.
class BaseEngine(ABC, metaclass=BaseInitMeta):

    def __init__(self):
        self.engine_name = "unknown"

        # Indicates if the engine can handle generators.
        self.can_consume_generators = False
        
        # Queue to manage tasks or data for the engine.
        self.queue = queue.Queue()

         # Callback to be called when an audio chunk is available.
        self.on_audio_chunk = None

         # Callback to be called when the engine is starting to synthesize audio.
        self.on_playback_start = None

    def get_stream_info(self):
        """
        Returns the audio stream configuration information suitable for PyAudio.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
                  - Format (int): The format of the audio stream. pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels. 1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz. 16000 represents 16kHz sample rate.
        """        
        raise NotImplementedError("The get_stream_info method must be implemented by the derived class.")

    def synthesize(self, 
                   text: str) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """
        raise NotImplementedError("The synthesize method must be implemented by the derived class.")
    
    def get_voices(self):
        """
        Retrieves the voices available from the specific voice source.

        This method should be overridden by the derived class to fetch the list of available voices.

        Returns:
            list: A list containing voice objects representing each available voice. 
        """
        raise NotImplementedError("The get_voices method must be implemented by the derived class.")
    
    def set_voice(self, voice: Union[str, object]):
        """
        Sets the voice to be used for speech synthesis.

        Args:
            voice (Union[str, object]): The voice to be used for speech synthesis.

        This method should be overridden by the derived class to set the desired voice.
        """
        raise NotImplementedError("The set_voice method must be implemented by the derived class.")
    
    def set_voice_parameters(self, **voice_parameters):
        """
        Sets the voice parameters to be used for speech synthesis.

        Args:
            **voice_parameters: The voice parameters to be used for speech synthesis.

        This method should be overridden by the derived class to set the desired voice parameters.
        """
        raise NotImplementedError("The set_voice_parameters method must be implemented by the derived class.")
    
    def shutdown(self):
        """
        Shuts down the engine.
        """
        pass

    def is_installed(self, lib_name: str) -> bool:
        """
        Check if the given library or software is installed and accessible.

        This method uses shutil.which to determine if the given library or software is
        installed and available in the system's PATH.

        Args:
            lib_name (str): Name of the library or software to check.

        Returns:
            bool: True if the library is installed, otherwise False.
        """        
        lib = shutil.which(lib_name)
        if lib is None:
            return False
        return True       