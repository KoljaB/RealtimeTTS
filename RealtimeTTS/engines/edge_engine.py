from .base_engine import BaseEngine
from edge_tts import Communicate, list_voices
import asyncio
from typing import Union
import queue
import threading
import subprocess
import pyaudio
import asyncio


class EdgeVoice:
    def __init__(self, name, full_name=None, gender=None, friendly_name=None, locale=None, status=None, suggested_codec=None, voice_tag=None):
        self.name = name
        self.full_name = full_name
        self.gender = gender
        self.friendly_name = friendly_name
        self.locale = locale
        self.status = status
        self.suggested_codec = suggested_codec
        self.voice_tag = voice_tag

    def __str__(self):
        # ANSI color codes for short format
        YELLOW = '\033[33m'
        RESET = '\033[0m'
        BLUE = '\033[38;2;51;153;255m'
        PINK = '\033[38;2;255;128;128m'

        return f"{YELLOW}{self.name}{RESET} ({BLUE if self.gender.lower() == 'male' else PINK}{self.gender}{RESET})"

    def __repr__(self):
        # ANSI color codes
        YELLOW = '\033[33m'
        RESET = '\033[0m'
        BLUE = '\033[38;2;51;153;255m'
        PINK = '\033[38;2;255;128;128m'

        # Format tags with proper indentation aligned with vertical bar
        tags = '\n'.join(f"  • {k}: {YELLOW}{v}{RESET}" for k, v in self.voice_tag.items())

        return f"""\
Name:   {YELLOW}{self.name}{RESET}
Gender: {BLUE if self.gender.lower() == 'male' else PINK}{self.gender}{RESET}
Full:   {YELLOW}{self.full_name}{RESET}
Human:  {YELLOW}{self.friendly_name}{RESET}
Locale: {YELLOW}{self.locale}{RESET}
Status: {YELLOW}{self.status}{RESET}
Codec:  {YELLOW}{self.suggested_codec}{RESET}
Tags:
{tags}
"""

class EdgeEngine(BaseEngine):
    def __init__(
        self,
        rate: int = 0,
        pitch: int = 0,        
        volume: int = 0,
   ):
        """
        Initializes a edge realtime text to speech engine object.
        """
        self.muted = False
        self.rate = rate
        self.pitch = pitch
        self.volume = volume
        self.on_playback_started = False
        self.current_voice = None

    def post_init(self):
        self.engine_name = "edge"

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration
        information suitable for Edge Engine.

        Returns:
            tuple: A tuple containing the audio format, number of channels,
              and the sample rate.
                  - Format (int): The format of the audio stream.
                    pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels.
                    1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz.
                    16000 represents 16kHz sample rate.
        """
        return pyaudio.paCustomFormat, -1, -1

    def synthesize(self, text):

        self.on_playback_started = False

        if self.current_voice is None:
            self.set_voice("en-US-EmmaMultilingualNeural")

        communicate = Communicate(
            text,
            self.current_voice.name,
            rate=f"{'+' if self.rate >= 0 else ''}{self.rate}%",
            volume=f"{'+' if self.volume >= 0 else ''}{self.volume}%",
            pitch=f"{'+' if self.pitch >= 0 else ''}{self.pitch}Hz",
            proxy=None,
        )        
        # Create queue for passing chunks between async and sync code
        chunk_queue = queue.Queue()
        
        # Function to run async stream in separate thread
        async def process_stream():
            try:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        self.queue.put(chunk["data"])
            except Exception as e:
                print(f"Stream processing error: {e}")
                chunk_queue.put(None)
                return False

        # Start async processing in thread
        def run_async_stream():
            asyncio.run(process_stream())

        thread = threading.Thread(target=run_async_stream)
        thread.start()
        thread.join()

        return True

    def get_voices(self):
        """
        Retrieves the installed voices available for the Coqui TTS engine.
        """
        voice_objects = []

        def get_voices_sync():
            async def get_sorted_voices():
                voices = await list_voices(proxy=None)
                return sorted(voices, key=lambda voice: voice["ShortName"])
            
            return asyncio.run(get_sorted_voices())                

        voices = get_voices_sync()
        for voice_data in voices:
            voice = EdgeVoice(
                name=voice_data['ShortName'],
                full_name=voice_data['Name'],
                gender=voice_data['Gender'],
                friendly_name=voice_data['FriendlyName'],
                locale=voice_data['Locale'],
                status=voice_data['Status'],
                suggested_codec=voice_data['SuggestedCodec'],
                voice_tag=voice_data['VoiceTag'],
            )
            voice_objects.append(voice)        

        return voice_objects

    def set_voice(self, voice: Union[str, EdgeVoice]):
        """
        Sets the voice to be used for speech synthesis.
        """
        if isinstance(voice, EdgeVoice):
            self.current_voice = voice
        else:
            installed_voices = self.get_voices()
            for installed_voice in installed_voices:
                if voice == installed_voice.name:
                    self.current_voice = installed_voice
                    return
            for installed_voice in installed_voices:
                if voice.lower() in installed_voice.name.lower():
                    self.current_voice = installed_voice
                    return

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets the voice parameters to be used for speech synthesis.

        Args:
            **voice_parameters: The voice parameters to be used for speech synthesis.

        This method should be overridden by the derived class to set the desired voice parameters.
        """
        pass

    def shutdown(self):
        """
        Shuts down the engine.
        """
        pass
