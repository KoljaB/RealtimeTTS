"""
Requires:
- pip install kokoro>=0.3.4 soundfile
- For non-English languages, install espeak-ng:
  !apt-get -qq -y install espeak-ng
"""

from .base_engine import BaseEngine
from queue import Queue
from typing import Optional, List
import numpy as np
import pyaudio
import time

# Import the new Kokoro pipeline
from kokoro import KPipeline

class KokoroEngine(BaseEngine):
    """
    A text-to-speech engine that uses the new Kokoro pipeline approach.
    It no longer relies on local .pt files or a specific model path. Instead,
    it uses KPipeline from the pip-installed `kokoro>=0.3.4`.
    """

    def __init__(
        self,
        default_lang_code: str = "a",
        default_voice: str = "af_heart",
        debug: bool = False
    ):
        """
        Initializes the Kokoro TTS engine.

        Args:
            default_lang_code (str): Language code to use (e.g., 'a' for American English).
            default_voice (str): Voice to use (e.g., 'af_heart').
            debug (bool): If True, prints debug info.
        """
        super().__init__()
        self.debug = debug
        self.engine_name = "kokoro"

        # Create a Kokoro pipeline. 
        # You can pass additional arguments if needed (e.g., use_cuda=True).
        self.pipeline = KPipeline(lang_code=default_lang_code)

        # Queue for feeding audio data to the output
        self.queue = Queue()

        # Keep track of current voice
        self.current_voice_name = default_voice

    def get_stream_info(self):
        """
        Returns PyAudio stream configuration for Kokoro audio.

        Returns:
            tuple: (format, channels, rate)
        """
        # Default sample rate for Kokoro is 24 kHz, mono, 16-bit
        return (pyaudio.paInt16, 1, 24000)

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text into audio data using Kokoro.

        Args:
            text (str): The text to be converted to speech.

        Returns:
            bool: True if successful, False otherwise.
        """
        start_time = time.time()

        try:
            # Use the pipeline to generate audio in chunks.
            # By default, the pipeline yields tuples: (graphemes, phonemes, audio_float32).
            # For multiline text, set split_pattern to something like r'\n+' if you want chunked output.
            generator = self.pipeline(text, voice=self.current_voice_name, speed=1.0)

            for index, (graphemes, phonemes, audio_float32) in enumerate(generator):
            # If audio_float32 is a Torch Tensor, convert it to NumPy
                if hasattr(audio_float32, "detach"):
                    audio_float32 = audio_float32.detach().cpu().numpy()

                # Convert float32 audio (range -1 to 1) to int16
                audio_int16 = (audio_float32 * 32767).astype(np.int16).tobytes()
                self.queue.put(audio_int16)

            if self.debug:
                duration = time.time() - start_time
                print(f"[KokoroEngine] Synthesis completed in {duration:.3f}s.")

            return True

        except Exception as e:
            print(f"[KokoroEngine] Error generating audio: {e}")
            return False

    def set_voice(self, voice_name: str):
        """
        Sets the voice used for speech synthesis.

        Args:
            voice_name (str): The new voice name (e.g., 'af_heart').
        """
        self.current_voice_name = voice_name
        if self.debug:
            print(f"[KokoroEngine] Voice set to: {voice_name}")

    def get_voices(self) -> List[str]:
        """
        Returns a list of all known Kokoro voice names, across languages.
        """
        return [
            # American English (lang_code='a')
            # Female (11)
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
            "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            # Male (9)
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
            "am_onyx", "am_puck", "am_santa",

            # British English (lang_code='b')
            # Female (4)
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            # Male (4)
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",

            # Japanese (lang_code='j')
            # Female (4)
            "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
            # Male (1)
            "jm_kumo",

            # Mandarin Chinese (lang_code='z')
            # Female (4)
            "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
            # Male (4)
            "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",

            # Spanish (lang_code='e')
            # Female (1)
            "ef_dora",
            # Male (2)
            "em_alex", "em_santa",

            # French (lang_code='f')
            # Female (1)
            "ff_siwis",

            # Hindi (lang_code='h')
            # Female (2)
            "hf_alpha", "hf_beta",
            # Male (2)
            "hm_omega", "hm_psi",

            # Italian (lang_code='i')
            # Female (1)
            "if_sara",
            # Male (1)
            "im_nicola",

            # Brazilian Portuguese (lang_code='p')
            # Female (1)
            "pf_dora",
            # Male (2)
            "pm_alex", "pm_santa",
        ]

    def shutdown(self):
        """
        Cleans up any resources used by KokoroEngine. 
        For the new pipeline, there's usually nothing special to do here.
        """
        pass
