from .base_engine import BaseEngine
from pydub import AudioSegment
from typing import Union
import tempfile
import wave
import os
import io
import pyaudio
from gtts import gTTS
import gtts.lang

SYNTHESIS_FILE = 'gtts_speech_synthesis.wav'


class GTTSVoice:
    def __init__(self, 
                 language: str = 'en',
                 tld: str = 'com',
                 chunk_length: int = 100,
                 crossfade_length: int = 10,
                 speed_increase: float = 1.0):
        self.language = language
        self.tld = tld
        self.chunk_length = chunk_length
        self.crossfade_length = crossfade_length
        self.speed_increase = speed_increase

    def __repr__(self):
        return f"{self.language} ({self.tld}), Chunk: {self.chunk_length}, Crossfade: {self.crossfade_length}, Speedup: {self.speed_increase}"


class GTTSEngine(BaseEngine):
    def __init__(self,
                 voice: Union[str, GTTSVoice] = GTTSVoice("en", "com"),
                 print_installed_voices: bool = False):
        """
        Initializes a gTTS text-to-speech engine object.

        Args:
            voice (Union[str, GTTSVoice], optional): Voice configuration. Defaults to GTTSVoice("en", "com").
            print_installed_voices (bool, optional): Indicates if the list of available languages should be printed. Defaults to False.
        """
        self.set_voice(voice)
        temp_file_path = tempfile.gettempdir()
        self.file_path = os.path.join(temp_file_path, SYNTHESIS_FILE)

        if print_installed_voices:
            print(self.get_voices())

    def post_init(self):
        self.engine_name = "gtts"

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration information suitable for gTTS Engine.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
                  - Format (int): The format of the audio stream. pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels. 1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz. 22050 represents 22.05kHz sample rate.
        """
        return pyaudio.paInt16, 1, 22050

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """
        try:
            # Generate audio with gTTS
            with io.BytesIO() as f:
                tts = gTTS(
                    text=text,
                    lang=self.voice.language,
                    tld=self.voice.tld)
                tts.write_to_fp(f)
                f.seek(0)
                
                audio = AudioSegment.from_file(f, format="mp3")

                if self.voice.speed_increase != 1.0:
                    audio = audio.speedup(
                        playback_speed=self.voice.speed_increase,
                        chunk_size=self.voice.chunk_length,
                        crossfade=self.voice.crossfade_length)

                audio.export(self.file_path, format="wav")

            # Now open the WAV file and read the chunks
            with wave.open(self.file_path, 'rb') as wf:
                audio_data = wf.readframes(wf.getnframes())
                self.queue.put(audio_data)
                return True

        except Exception as e:
            print(f"Error in synthesizing text: {e}")
            return False

    def get_voices(self):
        """
        Retrieves the voices available from gTTS.

        Returns:
            list[GTTSVoice]: A list containing GTTSVoice objects representing each available language and TLD.
        """
        voices = []
        languages = gtts.lang.tts_langs()
        tlds = ['com', 'com.au', 'co.uk', 'us', 'ca', 'co.in', 'ie', 'co.za']

        for lang in languages.keys():
            for tld in tlds:
                voices.append(GTTSVoice(language=lang, tld=tld))

        return voices

    def set_voice(self, voice: Union[str, GTTSVoice]):
        """
        Sets the voice to be used for speech synthesis.

        Args:
            voice (Union[str, GTTSVoice]): The voice to be used for speech synthesis.
        """
        if isinstance(voice, GTTSVoice):
            self.voice = voice
        else:
            self.voice = GTTSVoice(language=voice, tld = 'com')
