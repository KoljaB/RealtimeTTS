"""
Stream management
"""

from pydub import AudioSegment
import numpy as np
import threading
import resampy
import pyaudio
import logging
import queue
import time
import io


class AudioConfiguration:
    """
    Defines the configuration for an audio stream.
    """

    def __init__(
        self,
        format: int = pyaudio.paInt16,
        channels: int = 1,
        rate: int = 16000,
        output_device_index=None,
        muted: bool = False,
    ):
        """
        Args:
            format (int): Audio format, defaults to pyaudio.paInt16
            channels (int): Number of channels, defaults to 1 (mono)
            rate (int): Sample rate, defaults to 16000
            output_device_index (int): Output device index, defaults to None
        """
        self.format = format
        self.channels = channels
        self.rate = rate
        self.output_device_index = output_device_index
        self.muted = muted


class AudioStream:
    """
    Handles audio stream operations
    - opening, starting, stopping, and closing
    """

    def __init__(self, config: AudioConfiguration):
        """
        Args:
            config (AudioConfiguration): Object containing audio settings.
        """
        self.config = config
        self.stream = None
        self.pyaudio_instance = pyaudio.PyAudio()
        self.actual_sample_rate = 0

    def get_supported_sample_rates(self, device_index):
        """
        Test which standard sample rates are supported by the specified device.
        
        Args:
            device_index (int): The index of the audio device to test
            
        Returns:
            list: List of supported sample rates
        """
        standard_rates = [8000, 9600, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000]
        supported_rates = []
        
        device_info = self.pyaudio_instance.get_device_info_by_index(device_index)
        max_channels = device_info.get('maxOutputChannels')
        
        # Test each standard sample rate
        for rate in standard_rates:
            try:
                if self.pyaudio_instance.is_format_supported(
                    rate,
                    output_device=device_index,
                    output_channels=max_channels,
                    output_format=self.config.format,
                ):
                    supported_rates.append(rate)
            except:
                continue
        return supported_rates

    def _get_best_sample_rate(self, device_index, desired_rate):
        """
        Determines the best available sample rate for the device.
        
        Args:
            device_index: Index of the audio device
            desired_rate: Preferred sample rate
            
        Returns:
            int: Best available sample rate
        """
        try:
            # First determine the actual device index to use
            actual_device_index = (device_index if device_index is not None 
                                else self.pyaudio_instance.get_default_output_device_info()['index'])
            
            # Now use the actual_device_index for getting device info and supported rates
            device_info = self.pyaudio_instance.get_device_info_by_index(actual_device_index)
            supported_rates = self.get_supported_sample_rates(actual_device_index)
            
            # Check if desired rate is supported
            if desired_rate in supported_rates:
                return desired_rate
                
            # Find the highest supported rate that's lower than desired_rate
            lower_rates = [r for r in supported_rates if r <= desired_rate]
            if lower_rates:
                return max(lower_rates)
                
            # If no lower rates, get the lowest higher rate
            higher_rates = [r for r in supported_rates if r > desired_rate]
            if higher_rates:
                return min(higher_rates)
                
            # If nothing else works, return device's default rate
            return int(device_info.get('defaultSampleRate', 44100))
            
        except Exception as e:
            logging.warning(f"Error determining sample rate: {e}")
            return 44100  # Safe fallback

    def open_stream(self):
        """Opens an audio stream."""

        # check for mpeg format
        pyChannels = self.config.channels
        desired_rate = self.config.rate
        pyOutput_device_index = self.config.output_device_index

        # Determine the best sample rate
        best_rate = self._get_best_sample_rate(pyOutput_device_index, desired_rate)
        self.actual_sample_rate = best_rate

        if self.config.muted:
            logging.debug("Muted mode, no opening stream")

        else:
            if self.config.format == pyaudio.paCustomFormat:
                pyFormat = self.pyaudio_instance.get_format_from_width(2)
                logging.debug(
                    "Opening stream for mpeg audio chunks, "
                    f"pyFormat: {pyFormat}, pyChannels: {pyChannels}, "
                    f"pySampleRate: {best_rate}"
                )
            else:
                pyFormat = self.config.format
                logging.debug(
                    "Opening stream for wave audio chunks, "
                    f"pyFormat: {pyFormat}, pyChannels: {pyChannels}, "
                    f"pySampleRate: {best_rate}"
                )
            try:
                self.stream = self.pyaudio_instance.open(
                    format=pyFormat,
                    channels=pyChannels,
                    rate=best_rate,
                    output_device_index=pyOutput_device_index,
                    output=True,
                )
            except Exception as e:
                print(f"Error opening stream: {e}")

                # Get the number of available audio devices
                device_count = self.pyaudio_instance.get_device_count()

                print("Available Audio Devices:\n")

                # Iterate through all devices and print their details
                for i in range(device_count):
                    device_info = self.pyaudio_instance.get_device_info_by_index(i)
                    print(f"Device Index: {i}")
                    print(f"  Name: {device_info['name']}")
                    print(f"  Sample Rate (Default): {device_info['defaultSampleRate']} Hz")
                    print(f"  Max Input Channels: {device_info['maxInputChannels']}")
                    print(f"  Max Output Channels: {device_info['maxOutputChannels']}")
                    print(f"  Host API: {self.pyaudio_instance.get_host_api_info_by_index(device_info['hostApi'])['name']}")
                    print("\n")

                exit(0)

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
            self.stream = None

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
            timeout (float): Time (in seconds) to wait
              before raising a queue.Empty exception.

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

    def __init__(
        self,
        audio_buffer: queue.Queue,
        config: AudioConfiguration,
        on_playback_start=None,
        on_playback_stop=None,
        on_audio_chunk=None,
        muted=False,
    ):
        """
        Args:
            audio_buffer (queue.Queue): Queue to be used as the audio buffer.
            config (AudioConfiguration): Object containing audio settings.
            on_playback_start (Callable, optional): Callback function to be
              called at the start of playback. Defaults to None.
            on_playback_stop (Callable, optional): Callback function to be
              called at the stop of playback. Defaults to None.
        """
        self.buffer_manager = AudioBufferManager(audio_buffer)
        self.audio_stream = AudioStream(config)
        self.playback_active = False
        self.immediate_stop = threading.Event()
        self.pause_event = threading.Event()
        self.playback_thread = None
        self.on_playback_start = on_playback_start
        self.on_playback_stop = on_playback_stop
        self.on_audio_chunk = on_audio_chunk
        self.first_chunk_played = False
        self.muted = muted

    def _play_chunk(self, chunk):
        """
        Plays a chunk of audio data.

        Args:
            chunk: Chunk of audio data to be played.
        """

        # handle mpeg
        if self.audio_stream.config.format == pyaudio.paCustomFormat:
            # convert to pcm using pydub
            segment = AudioSegment.from_file(io.BytesIO(chunk), format="mp3")
            chunk = segment.raw_data

        if self.audio_stream.config.rate != self.audio_stream.actual_sample_rate:
            if self.audio_stream.config.format == pyaudio.paFloat32:
                audio_data = np.frombuffer(chunk, dtype=np.float32)
                resampled_data = resampy.resample(audio_data, self.audio_stream.config.rate, self.audio_stream.actual_sample_rate)
                chunk = resampled_data.astype(np.float32).tobytes()
            else:
                audio_data = np.frombuffer(chunk, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
                resampled_data = resampy.resample(audio_data, self.audio_stream.config.rate, self.audio_stream.actual_sample_rate)
                chunk = (resampled_data * 32768.0).astype(np.int16).tobytes()

        sub_chunk_size = 256

        for i in range(0, len(chunk), sub_chunk_size):
            sub_chunk = chunk[i : i + sub_chunk_size]

            if not self.first_chunk_played and self.on_playback_start:
                self.on_playback_start()
                self.first_chunk_played = True

            if not self.muted:
                try:
                    # Wait until there's space in the buffer to write the next chunk
                    while self.audio_stream.stream.get_write_available() < len(sub_chunk):
                        time.sleep(0.001)  # Small sleep to let the stream process audio

                    self.audio_stream.stream.write(sub_chunk)
                except Exception as e:
                    print(f"RealtimeTTS error sending audio data: {e}")

            if self.on_audio_chunk:
                self.on_audio_chunk(sub_chunk)

            # Pause playback if the event is set
            while self.pause_event.is_set():
                time.sleep(0.01)

            if self.immediate_stop.is_set():
                break


    def _process_buffer(self):
        """
        Processes and plays audio data from the buffer
        until it's empty or playback is stopped.
        """
        while self.playback_active or not self.buffer_manager.audio_buffer.empty():
            chunk = self.buffer_manager.get_from_buffer()
            if chunk:
                self._play_chunk(chunk)

            if self.immediate_stop.is_set():
                logging.info("Immediate stop requested, aborting playback")
                break

        if self.on_playback_stop:
            self.on_playback_stop()

    def get_buffered_seconds(self) -> float:
        """
        Calculates the duration (in seconds) of the buffered audio data.

        Returns:
            float: Duration of buffered audio in seconds.
        """
        total_samples = sum(
            len(chunk) // 2 for chunk in list(self.buffer_manager.audio_buffer.queue)
        )
        return total_samples / self.audio_stream.config.rate

    def start(self):
        """Starts audio playback."""
        self.first_chunk_played = False
        self.playback_active = True
        if not self.audio_stream.stream:
            self.audio_stream.open_stream()

        self.audio_stream.start_stream()

        if not self.playback_thread or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(target=self._process_buffer)
            self.playback_thread.start()

    def stop(self, immediate: bool = False):
        """
        Stops audio playback.

        Args:
            immediate (bool): If True, stops playback immediately
              without waiting for buffer to empty.
        """
        if not self.playback_thread:
            logging.warn("No playback thread found, cannot stop playback")
            return

        if immediate:
            self.immediate_stop.set()
            while self.playback_active:
                time.sleep(0.1)
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

    def mute(self, muted: bool = True):
        """Mutes audio playback."""
        self.muted = muted
