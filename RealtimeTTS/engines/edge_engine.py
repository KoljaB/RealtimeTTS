from .base_engine import BaseEngine
from edge_tts import Communicate, list_voices
import asyncio
import queue
import threading
import subprocess
import pyaudio
import asyncio


class EdgeVoice:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"{self.name}"


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
        self.can_playout = True
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

        self.on_playback_started = False
        print(f"Synthesizing text: {text}")

        if self.current_voice is None:
            self.set_voice("en-US-EmmaMultilingualNeural")

        communicate = Communicate(
            text,
            self.current_voice.name,
            rate=f"+{self.rate}%",
            volume=f"+{self.volume}%",
            pitch=f"+{self.pitch}Hz",
            proxy=None,
        )        
        # Create queue for passing chunks between async and sync code
        chunk_queue = queue.Queue()
        
        # Function to run async stream in separate thread
        async def process_stream():
            try:
                async for chunk in communicate.stream():
                    chunk_queue.put(chunk)
                chunk_queue.put(None)  # Signal end of stream
            except Exception as e:
                print(f"Stream processing error: {e}")
                chunk_queue.put(None)
                return False

        # Start async processing in thread
        def run_async_stream():
            asyncio.run(process_stream())

        thread = threading.Thread(target=run_async_stream)
        thread.start()

        # Process chunks as they arrive
        try:
            while True:
                chunk = chunk_queue.get()
                if chunk is None:
                    break
                if chunk["type"] == "audio":
                    audio_chunk = chunk["data"]

                    if not self.on_playback_started and self.on_playback_start:
                        self.on_playback_start()
                    self.on_playback_started = True
                    if not self.muted:
                        self.mpv_process.stdin.write(audio_chunk)
                        self.mpv_process.stdin.flush()
                    if self.on_audio_chunk:
                        self.on_audio_chunk(audio_chunk)
            return True
        except Exception as e:
            print(f"Chunk processing error: {e}")
            return False
        finally:
            thread.join()

            if not self.muted:
                if self.mpv_process.stdin:
                    self.mpv_process.stdin.close()
                self.mpv_process.wait()            

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
            )
            voice_objects.append(voice)        

        return voice_objects

    def set_voice(self, voice: str):
        """
        Sets the voice to be used for speech synthesis.
        """
        if isinstance(voice, EdgeVoice):
            self.current_voice = voice
        else:
            installed_voices = self.get_voices()
            for installed_voice in installed_voices:
                if voice in installed_voice.name:
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
