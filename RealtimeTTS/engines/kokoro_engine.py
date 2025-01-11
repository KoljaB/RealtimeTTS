"""
Needs:
- pip install munch

"""
from .base_engine import BaseEngine
from typing import Optional, Dict
from queue import Queue
import pyaudio
import torch
import time
import sys
import os

# You may need these if you plan to write WAV files or do numeric conversions
import numpy as np
# from scipy.io.wavfile import write  # Uncomment if you want to save WAV files

class KokoroEngine(BaseEngine):
    """
    A simple TTS engine that uses the Kokoro model for voice synthesis.
    Loads all voices on init, allows setting a current voice, and generates audio.
    """

    def __init__(
        self,
        kokoro_root: str, 
        model_path: str = "kokoro-v0_19.pth",
        voice_names: Optional[list] = None,
        voices_dir: str = "voices",
        debug: bool = False
    ):
        """
        Initializes the Kokoro text-to-speech engine.

        Args:
            model_path (str): Path to the Kokoro model checkpoint.
            voice_names (list): List of voice names you want to load. Defaults to a set of known voices.
            voices_dir (str): Directory where voice .pt files are stored.
            debug (bool): If True, prints debug info.
        """
        super().__init__()  # Ensure BaseEngine is properly initialized
        self.debug = debug
        self.kokoro_root = kokoro_root.replace("\\", "/")

        # Add the root directory to sys.path
        root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), self.kokoro_root)) 
        if self.debug:
            print(f"Adding {root_directory} to sys.path")
        sys.path.append(root_directory)

        self.queue = Queue()  # Queue for feeding audio data to the output
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build the main model once
        from models import build_model  # Kokoro-specific import
        if not os.path.exists(model_path):
            model_path = os.path.join(self.kokoro_root, model_path)
        self.model = build_model(model_path, self.device)
        if self.debug:
            print(f"Kokoro model loaded from: {model_path} (device: {self.device})")

        # If user didn't provide a voice list, fall back to defaults
        if voice_names is None:
            # This is just an example set; customize as needed
            voice_names = [
                "af_nicole", 
                "af",
                "af_bella",
                "af_sarah",
                "am_adam",
                "am_michael",
                "bf_emma",
                "bf_isabella",
                "bm_george",
                "bm_lewis",
                "af_sky"
            ]
        self.voicepacks_dir = voices_dir
        self.voicepacks: Dict[str, torch.nn.Module] = {}
        self._load_voices(voice_names)

        # Pick the first voice as current by default (or None)
        self.current_voice_name = voice_names[0] if voice_names else None
        self.current_voicepack = self.voicepacks.get(self.current_voice_name, None)

        # Warm up the model if possible
        self._warm_up_model()

        self.post_init()

    def post_init(self):
        """
        Called after initialization. Sets the engine name.
        """
        self.engine_name = "kokoro"

    def _load_voices(self, voice_names: list):
        """
        Loads all specified voice .pt files into memory and stores them in a dict.
        """
        for voice_name in voice_names:
            try:
                path = os.path.join(self.voicepacks_dir, f"{voice_name}.pt")
                if not os.path.exists(path):
                    path = os.path.join(self.kokoro_root, path)
                voicepack = torch.load(path, weights_only=True).to(self.device)
                self.voicepacks[voice_name] = voicepack
                if self.debug:
                    print(f"Loaded Kokoro voice: {voice_name}")
            except Exception as e:
                print(f"Failed to load voice {voice_name} from {path}. Error: {e}")

    def _warm_up_model(self):
        from kokoro import generate  # Kokoro-specific import

        """
        Runs a quick, minimal synthesis to get everything ready.
        """
        if self.current_voicepack is None:
            print("No voice is currently set. Skipping model warm-up.")
            return 

        warm_text = "Hello world."
        if self.debug:
            print(f"Warming up model with voice: {self.current_voice_name}")
        try:
            # We only care that it runs without error
            generate(self.model, warm_text, self.current_voicepack, lang=self.current_voice_name[0])
            if self.debug:
                print("Kokoro model warm-up completed.")
        except Exception as e:
            print(f"Warning: Model warm-up failed. {e}")

    def get_stream_info(self):
        """
        Returns PyAudio stream configuration for Kokoro audio.

        Returns:
            tuple: (format, channels, rate)
        """
        # Kokoro examples use a 24 kHz sample rate
        return (pyaudio.paInt16, 1, 24000)

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text into audio data using Kokoro.

        Args:
            text (str): The text to be converted to speech.

        Returns:
            bool: True if successful, False otherwise.
        """
        from kokoro import generate  # Kokoro-specific import

        if self.current_voicepack is None:
            print("No valid voice is currently set.")
            return False

        start_time = time.time()

        try:
            # The lang argument is just the first character of the voice name in the example
            lang_code = self.current_voice_name[0] if self.current_voice_name else "a"

            # Generate float32 audio with Kokoro (assumption)
            audio_float32, _ = generate(self.model, text, self.current_voicepack, lang=lang_code)

            # Convert to int16 for playback
            audio_int16 = (audio_float32 * 32767).astype(np.int16).tobytes()

            # Put the audio in our queue
            self.queue.put(audio_int16)

            if self.debug:
                end_time = time.time()
                print(f"Synthesis complete in {end_time - start_time:.3f}s.")

            return True

        except Exception as e:
            print(f"Error generating audio: {e}")
            return False

    def set_voice(self, voice_name: str):
        """
        Sets the voice used for speech synthesis to one of the loaded voicepacks.

        Args:
            voice_name (str): The name of the voice pack (e.g., 'af_sarah').
        """
        if voice_name in self.voicepacks:
            self.current_voice_name = voice_name
            self.current_voicepack = self.voicepacks[voice_name]
            if self.debug:
                print(f"Voice set to {voice_name}")
        else:
            print(f"Voice '{voice_name}' not found in loaded voicepacks.")

    def get_voices(self):
        """
        Returns a list of loaded voice names.

        Returns:
            list[str]: The loaded voice names.
        """
        return list(self.voicepacks.keys())

    def shutdown(self):
        """
        Cleans up any resources used by KokoroEngine.
        """
        # If there's anything to release or finalize, do it here.
        pass
