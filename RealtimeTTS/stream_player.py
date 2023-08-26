"""
Stream management
"""

import pyaudio
import queue
import threading
import logging
import time


class AudioConfiguration:
    """
    Defines the configuration for an audio stream.
    """

    def __init__(self, format: int = pyaudio.paInt16, channels: int = 1, rate: int = 16000):
        """
        Args:
            format (int): Audio format, defaults to pyaudio.paInt16
            channels (int): Number of channels, defaults to 1 (mono)
            rate (int): Sample rate, defaults to 16000
        """
        self.format = format
        self.channels = channels
        self.rate = rate


class AudioStream:
    """
    Handles audio stream operations such as opening, starting, stopping, and closing.
    """

    def __init__(self, config: AudioConfiguration):
        """
        Args:
            config (AudioConfiguration): Object containing audio settings.
        """
        self.config = config
        self.stream = None
        self.pyaudio_instance = pyaudio.PyAudio()

    def open_stream(self):
        """Opens an audio stream."""
        self.stream = self.pyaudio_instance.open(format=self.config.format, channels=self.config.channels, rate=self.config.rate, output=True)

    def start_stream(self):
        """Starts the audio stream."""
        if self.stream and not self.stream.is_active():
            self.stream.start_stream()

    def stop_stream(self):
        """Stops the audio stream."""
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()

    def close_stream(self):
        """Closes the audio stream."""
        if self.stream:
            self.stop_stream()
            self.stream.close()

    def is_stream_active(self) -> bool:
        """
        Checks if the audio stream is active.

        Returns:
            bool: True if the stream is active, False otherwise.
        """
        return self.stream and self.stream.is_active()


class AudioBufferManager:
    """
    Manages an audio buffer, allowing addition and retrieval of audio data.
    """

    def __init__(self, audio_buffer: queue.Queue):
        """
        Args:
            audio_buffer (queue.Queue): Queue to be used as the audio buffer.
        """
        self.audio_buffer = audio_buffer
        self.total_samples = 0

    def add_to_buffer(self, audio_data):
        """
        Adds audio data to the buffer.

        Args:
            audio_data: Audio data to be added.
        """
        self.audio_buffer.put(audio_data)
        self.total_samples += len(audio_data) // 2

    def clear_buffer(self):
        """Clears all audio data from the buffer."""
        while not self.audio_buffer.empty():
            try:
                self.audio_buffer.get_nowait()
            except queue.Empty:
                continue
        self.total_samples = 0

    def get_from_buffer(self, timeout: float = 0.05):
        """
        Retrieves audio data from the buffer.

        Args:
            timeout (float): Time (in seconds) to wait before raising a queue.Empty exception.

        Returns:
            The audio data chunk or None if the buffer is empty.
        """
        try:
            chunk = self.audio_buffer.get(timeout=timeout)
            self.total_samples -= len(chunk) // 2
            return chunk
        except queue.Empty:
            return None

    def get_buffered_seconds(self, rate: int) -> float:
        """
        Calculates the duration (in seconds) of the buffered audio data.

        Args:
            rate (int): Sample rate of the audio data.

        Returns:
            float: Duration of buffered audio in seconds.
        """
        return self.total_samples / rate


class StreamPlayer:
    """
    Manages audio playback operations such as start, stop, pause, and resume.
    """

    def __init__(self, audio_buffer: queue.Queue, config: AudioConfiguration):
        """
        Args:
            audio_buffer (queue.Queue): Queue to be used as the audio buffer.
            config (AudioConfiguration): Object containing audio settings.
        """
        self.buffer_manager = AudioBufferManager(audio_buffer)
        self.audio_stream = AudioStream(config)
        self.playback_active = False
        self.immediate_stop = threading.Event()
        self.pause_event = threading.Event()
        self.playback_thread = None

    def _play_chunk(self, chunk):
        """
        Plays a chunk of audio data.

        Args:
            chunk: Chunk of audio data to be played.
        """
        sub_chunk_size = 1024
        for i in range(0, len(chunk), sub_chunk_size):
            sub_chunk = chunk[i:i + sub_chunk_size]
            self.audio_stream.stream.write(sub_chunk)

            # Pause playback if the event is set
            while self.pause_event.is_set():
                time.sleep(0.01)

            if self.immediate_stop.is_set():
                break

    def _process_buffer(self):
        """Processes and plays audio data from the buffer until it's empty or playback is stopped."""
        while self.playback_active or not self.buffer_manager.audio_buffer.empty():
            chunk = self.buffer_manager.get_from_buffer()
            if chunk:
                self._play_chunk(chunk)

            if self.immediate_stop.is_set():
                logging.info("Immediate stop requested, aborting playback")
                break

    def get_buffered_seconds(self) -> float:
        """
        Calculates the duration (in seconds) of the buffered audio data.

        Returns:
            float: Duration of buffered audio in seconds.
        """
        total_samples = sum(len(chunk) // 2 for chunk in list(self.buffer_manager.audio_buffer.queue))
        return total_samples / self.audio_stream.config.rate

    def start(self):
        """Starts audio playback."""
        self.playback_active = True
        self.audio_stream.open_stream()
        self.audio_stream.start_stream()
        self.playback_thread = threading.Thread(target=self._process_buffer)
        self.playback_thread.start()

    def stop(self, immediate: bool = False):
        """
        Stops audio playback.

        Args:
            immediate (bool): If True, stops playback immediately without waiting for buffer to empty.
        """
        if not self.playback_thread:
            logging.warn("No playback thread found, cannot stop playback")
            return

        if immediate:
            self.immediate_stop.set()
            return

        self.playback_active = False

        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join()

        time.sleep(0.1)

        self.audio_stream.close_stream()
        self.immediate_stop.clear()
        self.buffer_manager.clear_buffer()
        self.playback_thread = None

    def pause(self):
        """Pauses audio playback."""
        self.pause_event.set()

    def resume(self):
        """Resumes paused audio playback."""
        self.pause_event.clear()


# import pyaudio
# import queue
# import threading
# import logging
# import time

# class AudioConfiguration:
#     def __init__(self, format: int = pyaudio.paInt16, channels: int = 1, rate: int = 16000):
#         self.format = format
#         self.channels = channels
#         self.rate = rate


# class AudioStream:
#     def __init__(self, config: AudioConfiguration):
#         self.config = config
#         self.stream = None
#         self.pyaudio_instance = pyaudio.PyAudio()

#     def open_stream(self):
#         self.stream = self.pyaudio_instance.open(format=self.config.format, channels=self.config.channels, rate=self.config.rate, output=True)
    
#     def start_stream(self):
#         if self.stream and not self.stream.is_active():
#             self.stream.start_stream()

#     def stop_stream(self):
#         if self.stream and self.stream.is_active():
#             self.stream.stop_stream()

#     def close_stream(self):
#         if self.stream:
#             self.stop_stream()
#             self.stream.close()

#     def is_stream_active(self):
#         return self.stream and self.stream.is_active()


# class AudioBufferManager:
#     def __init__(self, audio_buffer: queue.Queue):
#         self.audio_buffer = audio_buffer
#         self.total_samples = 0

#     def add_to_buffer(self, audio_data):
#         self.audio_buffer.put(audio_data)
#         self.total_samples += len(audio_data) // 2

#     def clear_buffer(self):
#         while not self.audio_buffer.empty():
#             try:
#                 self.audio_buffer.get_nowait()
#             except queue.Empty:
#                 continue
#         self.total_samples = 0        

#     def get_from_buffer(self, timeout: float = 0.05):
#         try:
#             chunk = self.audio_buffer.get(timeout=timeout)
#             self.total_samples -= len(chunk) // 2
#             return chunk
#         except queue.Empty:
#             return None

#     def get_buffered_seconds(self, rate: int) -> float:
#         return self.total_samples / rate


# class StreamPlayer:
#     def __init__(self, audio_buffer: queue.Queue, config: AudioConfiguration):
#         self.buffer_manager = AudioBufferManager(audio_buffer)
#         self.audio_stream = AudioStream(config)
#         self.playback_active = False
#         self.immediate_stop = threading.Event()
#         self.pause_event = threading.Event()
#         self.playback_thread = None

#     def _play_chunk(self, chunk):
#         sub_chunk_size = 1024
#         for i in range(0, len(chunk), sub_chunk_size):
#             sub_chunk = chunk[i:i + sub_chunk_size]
#             self.audio_stream.stream.write(sub_chunk)

#             # Pause playback if the event is set
#             while self.pause_event.is_set():
#                 time.sleep(0.01)
            
#             if self.immediate_stop.is_set():
#                 break

#     def _process_buffer(self):
#         while self.playback_active or not self.buffer_manager.audio_buffer.empty():
#             chunk = self.buffer_manager.get_from_buffer()
#             if chunk:
#                 self._play_chunk(chunk)

#             if self.immediate_stop.is_set():
#                 logging.info("Immediate stop requested, aborting playback")
#                 break

#     def get_buffered_seconds(self):
#         total_samples = sum(len(chunk) // 2 for chunk in list(self.buffer_manager.audio_buffer.queue))
#         return total_samples / self.audio_stream.config.rate  

#     def start(self):
#         self.playback_active = True
#         self.audio_stream.open_stream()
#         self.audio_stream.start_stream()
#         self.playback_thread = threading.Thread(target=self._process_buffer)
#         self.playback_thread.start()

#     def stop(self, immediate: bool = False):

#         if not self.playback_thread:
#             logging.warn("No playback thread found, cannot stop playback")
#             return

#         if immediate:
#             self.immediate_stop.set()
#             return

#         self.playback_active = False
        
#         if self.playback_thread and self.playback_thread.is_alive():
#             self.playback_thread.join()

#         time.sleep(0.1)

#         self.audio_stream.close_stream()
#         self.immediate_stop.clear()
#         self.buffer_manager.clear_buffer()
#         self.playback_thread = None

#     def pause(self):
#         self.pause_event.set()

#     def resume(self):
#         self.pause_event.clear()
