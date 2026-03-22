"""
This module defines a base framework for speech synthesis engines. It includes:
- A TimingInfo class to capture timing details (start, end, and word) of audio segments.
- A BaseEngine abstract class (using a custom metaclass) that sets up default properties and common audio processing methods (such as applying fade-ins/outs and trimming silence) along with abstract methods for voice management and synthesis.
"""

import torch.multiprocessing as mp
from abc import ABCMeta, ABC
from typing import Union
import numpy as np
import shutil
import queue

class TimingInfo:
    def __init__(self, start_time, end_time, word):
        self.start_time = start_time
        self.end_time = end_time
        self.word = word

    def __str__(self):
        return f"Word: {self.word}, Start Time: {self.start_time}, End Time: {self.end_time}"

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

        # Queue to manage audio chunks for the engine.
        self.queue = queue.Queue()

        # Queue to manage word level timings for the engine.
        self.timings = queue.Queue()

        # Callback to be called when an audio chunk is available.
        self.on_audio_chunk = None

        # Callback to be called when the engine is starting to synthesize audio.
        self.on_playback_start = None

        self.stop_synthesis_event = mp.Event()

        self.reset_audio_duration()

    def reset_audio_duration(self):
        """
        Resets the audio duration to 0.
        """
        self.audio_duration = 0

    def apply_fade_in(self, audio: np.ndarray, sample_rate: int = -1, fade_duration_ms: int = 15) -> np.ndarray:
        """
        Applies a linear fade-in over fade_duration_ms at the start of the audio.
        """
        sample_rate = self.verify_sample_rate(sample_rate)
        audio = audio.copy()

        fade_samples = int(sample_rate * fade_duration_ms / 1000)
        if fade_samples == 0 or len(audio) < fade_samples:
            fade_samples = len(audio)
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        audio[:fade_samples] *= fade_in
        return audio

    def apply_fade_out(self, audio: np.ndarray, sample_rate: int = -1, fade_duration_ms: int = 15) -> np.ndarray:
        """
        Applies a linear fade-out over fade_duration_ms at the end of the audio.
        """
        sample_rate = self.verify_sample_rate(sample_rate)
        audio = audio.copy()

        fade_samples = int(sample_rate * fade_duration_ms / 1000)
        if fade_samples == 0 or len(audio) < fade_samples:
            fade_samples = len(audio)
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        audio[-fade_samples:] *= fade_out
        return audio

    def trim_silence_start(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 24000,
        silence_threshold: float = 0.01,
        extra_ms: int = 25,
        fade_in_ms: int = 15
    ) -> np.ndarray:
        """
        Removes leading silence from audio_data, applies extra trimming, and fades-in if trimming occurred.

        Args:
            audio_data (np.ndarray): The audio data to process.
            sample_rate (int): The sample rate of the audio data.
            silence_threshold (float): The threshold for silence detection.
            extra_ms (int): Additional milliseconds to trim from the start.
            fade_in_ms (int): Milliseconds for fade-in effect.
        """
        sample_rate = self.verify_sample_rate(sample_rate)
        trimmed = False
        audio_data = audio_data.copy()
        non_silent = np.where(np.abs(audio_data) > silence_threshold)[0]
        if len(non_silent) > 0:
            start_index = non_silent[0]
            if start_index > 0:
                trimmed = True
            audio_data = audio_data[start_index:]

        extra_samples = int(extra_ms * sample_rate / 1000)
        if extra_samples > 0 and len(audio_data) > extra_samples:
            audio_data = audio_data[extra_samples:]
            trimmed = True

        if trimmed:
            audio_data = self.apply_fade_in(audio_data, sample_rate, fade_in_ms)
        return audio_data

    def trim_silence_end(
        self,
        audio_data: np.ndarray,
        sample_rate: int = -1,
        silence_threshold: float = 0.01,
        extra_ms: int = 50,
        fade_out_ms: int = 15
    ) -> np.ndarray:
        """
        Removes trailing silence from audio_data, applies extra trimming, and fades-out if trimming occurred.

        Args:
            audio_data (np.ndarray): The audio data to be trimmed.
            sample_rate (int): The sample rate of the audio data. Default is -1.
            silence_threshold (float): The threshold below which audio is considered silent. Default is 0.01.
            extra_ms (int): Extra milliseconds to trim from the end of the audio. Default is 50.
            fade_out_ms (int): Milliseconds for fade-out effect at the end of the audio. Default is 15.
        """
        sample_rate = self.verify_sample_rate(sample_rate)
        trimmed = False
        audio_data = audio_data.copy()
        non_silent = np.where(np.abs(audio_data) > silence_threshold)[0]
        if len(non_silent) > 0:
            end_index = non_silent[-1] + 1
            if end_index < len(audio_data):
                trimmed = True
            audio_data = audio_data[:end_index]

        extra_samples = int(extra_ms * sample_rate / 1000)
        if extra_samples > 0 and len(audio_data) > extra_samples:
            audio_data = audio_data[:-extra_samples]
            trimmed = True

        if trimmed:
            audio_data = self.apply_fade_out(audio_data, sample_rate, fade_out_ms)
        return audio_data

    def verify_sample_rate(self, sample_rate: int) -> int:
        """
        Verifies and returns the sample rate.
        If the sample rate is -1, it will be obtained from the engine's configuration.
        """
        if sample_rate == -1:
            _, _, sample_rate = self.get_stream_info()
            if sample_rate == -1:
                raise ValueError("Sample rate must be provided or obtained from get_stream_info.")
        return sample_rate

    def _trim_silence(
        self,
        audio_data: np.ndarray,
        sample_rate: int = -1,
        silence_threshold: float = 0.005,
        extra_start_ms: int = 15,
        extra_end_ms: int = 15,
        fade_in_ms: int = 10,
        fade_out_ms: int = 10
    ) -> np.ndarray:
        """
        Removes silence from both the start and end of audio_data.
        If trimming occurs on either end, the corresponding fade is applied.
        """
        sample_rate = self.verify_sample_rate(sample_rate)

        audio_data = self.trim_silence_start(
            audio_data, sample_rate, silence_threshold, extra_start_ms, fade_in_ms
        )
        audio_data = self.trim_silence_end(
            audio_data, sample_rate, silence_threshold, extra_end_ms, fade_out_ms
        )
        return audio_data


    def get_stream_info(self):
        """
        Returns the audio stream configuration information suitable for PyAudio.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
                  - Format (int): The format of the audio stream. pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels. 1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz. 16000 represents 16kHz sample rate.
        """
        raise NotImplementedError(
            "The get_stream_info method must be implemented by the derived class."
        )

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """
        self.stop_synthesis_event.clear()

    def get_voices(self):
        """
        Retrieves the voices available from the specific voice source.

        This method should be overridden by the derived class to fetch the list of available voices.

        Returns:
            list: A list containing voice objects representing each available voice.
        """
        raise NotImplementedError(
            "The get_voices method must be implemented by the derived class."
        )

    def set_voice(self, voice: Union[str, object]):
        """
        Sets the voice to be used for speech synthesis.

        Args:
            voice (Union[str, object]): The voice to be used for speech synthesis.

        This method should be overridden by the derived class to set the desired voice.
        """
        raise NotImplementedError(
            "The set_voice method must be implemented by the derived class."
        )

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets the voice parameters to be used for speech synthesis.

        Args:
            **voice_parameters: The voice parameters to be used for speech synthesis.

        This method should be overridden by the derived class to set the desired voice parameters.
        """
        raise NotImplementedError(
            "The set_voice_parameters method must be implemented by the derived class."
        )

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

    def stop(self):
        """
        Stops the engine.
        """
        self.stop_synthesis_event.set()
