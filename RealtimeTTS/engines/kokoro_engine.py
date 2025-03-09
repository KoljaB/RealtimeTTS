"""
Requires:
- pip install kokoro>=0.3.4 soundfile
- For non-English languages, install espeak-ng:
  !apt-get -qq -y install espeak-ng
"""

from .base_engine import BaseEngine
from queue import Queue
from typing import List
import numpy as np
import pyaudio
import time

# Import the text-to-speech pipeline from the Kokoro package.
from kokoro import KPipeline

class KokoroEngine(BaseEngine):
    """
    A text-to-speech (TTS) engine utilizing the Kokoro pipeline.

    This engine supports multiple languages by caching separate pipeline instances
    for each language code. The engine selects the appropriate pipeline based on
    the voice's prefix, allowing it to manage voice and language settings dynamically.

    Attributes:
        debug (bool): Flag to enable or disable debug messages.
        engine_name (str): Identifier for the engine (set to "kokoro").
        queue (Queue): Stores audio chunks for playback.
        pipelines (dict): Caches KPipeline instances keyed by language code.
        current_voice_name (str): Currently selected voice identifier.
        current_lang (str): Language code derived from the current voice.
        speed (float): Speed factor for speech synthesis.
    """

    def __init__(
            self,
            default_lang_code: str = "a",
            default_voice: str = "af_heart",
            default_speed: float = 1.0,
            debug: bool = False):
        """
        Initializes the KokoroEngine with default settings.

        This sets up the audio queue, voice and language selection, and creates
        the initial pipeline for the default language.

        Args:
            default_lang_code (str): Fallback language code if the voice doesn't specify one.
            default_voice (str): Default voice to use (e.g., "af_heart").
            default_speed (float): Default speed factor for speech synthesis.
            debug (bool): If True, prints detailed debug output.
        """
        super().__init__()
        self.debug = debug
        self.engine_name = "kokoro"
        self.queue = Queue()  # Queue for streaming audio data.
        self.pipelines = {}   # Cache pipelines based on language code.

        # Default speed parameter for synthesis.
        self.speed = default_speed

        self.current_voice_name = default_voice
        # Attempt to derive the language code from the voice name.
        lang_from_voice = self._get_lang_code_from_voice(default_voice)
        self.current_lang = lang_from_voice if lang_from_voice else default_lang_code

        # Create and cache the pipeline for the current language.
        self.pipelines[self.current_lang] = KPipeline(lang_code=self.current_lang)

        if self.debug:
            print(f"[KokoroEngine] Initialized with voice: {self.current_voice_name} (lang: {self.current_lang}), speed: {self.speed}")


    def _get_lang_code_from_voice(self, voice_name: str) -> str:
        """
        Determines the language code from the provided voice name.

        This method checks for specific prefixes in the voice name and maps them
        to a single-character language code.

        Args:
            voice_name (str): The voice identifier (e.g., "af_heart").

        Returns:
            str: The language code corresponding to the voice prefix, or None if unknown.
        """
        # Map voice name prefixes to language codes.
        if voice_name.startswith(("af_", "am_")):
            return "a"  # American English
        elif voice_name.startswith(("bf_", "bm_")):
            return "b"  # British English
        elif voice_name.startswith(("jf_", "jm_")):
            return "j"  # Japanese
        elif voice_name.startswith(("zf_", "zm_")):
            return "z"  # Mandarin Chinese
        elif voice_name.startswith(("ef_", "em_")):
            return "e"  # Spanish
        elif voice_name.startswith("ff_"):
            return "f"  # French
        elif voice_name.startswith(("hf_", "hm_")):
            return "h"  # Hindi
        elif voice_name.startswith(("if_", "im_")):
            return "i"  # Italian
        elif voice_name.startswith(("pf_", "pm_")):
            return "p"  # Brazilian Portuguese
        else:
            return None

    def _get_pipeline(self, lang_code: str):
        """
        Retrieves the KPipeline for the specified language code.

        If the pipeline does not exist in the cache, it is created and stored.

        Args:
            lang_code (str): The language code for which to get the pipeline.

        Returns:
            KPipeline: The pipeline instance corresponding to the language code.
        """
        if lang_code not in self.pipelines:
            if self.debug:
                print(f"[KokoroEngine] Creating new pipeline for language code: {lang_code}")
            self.pipelines[lang_code] = KPipeline(lang_code=lang_code)
        return self.pipelines[lang_code]

    def get_stream_info(self):
        """
        Provides the PyAudio stream configuration for the synthesized audio.

        Returns:
            tuple: A tuple (format, channels, rate) where:
                - format: Audio format (e.g., pyaudio.paInt16),
                - channels: Number of audio channels (1 for mono),
                - rate: Sample rate in Hz (24000 for Kokoro).
        """
        # Kokoro uses 24 kHz sampling rate, mono channel, with 16-bit samples.
        return (pyaudio.paInt16, 1, 24000)

    def synthesize(self, text: str) -> bool:
        """
        Converts the input text into speech audio.

        The method uses the appropriate pipeline for the current language to generate
        audio data in chunks. Each chunk is processed to convert from float32 to int16
        format and then added to the audio queue.

        Args:
            text (str): The text string to synthesize.

        Returns:
            bool: True if synthesis is successful, False otherwise.
        """
        start_time = time.time()
        try:
            if self.debug:
                print(f"[KokoroEngine] Synthesizing with language code: {self.current_lang} and speed: {self.speed}")
            # Get or create the pipeline corresponding to the current language.
            pipeline = self._get_pipeline(self.current_lang)
            # Generate audio in chunks from the pipeline using the speed parameter.
            generator = pipeline(text, voice=self.current_voice_name, speed=self.speed)

            for index, (graphemes, phonemes, audio_float32) in enumerate(generator):
                # If audio is provided as a Torch Tensor, convert it to a NumPy array.
                if hasattr(audio_float32, "detach"):
                    audio_float32 = audio_float32.detach().cpu().numpy()
                # Convert float32 audio (range -1.0 to 1.0) to int16 format.
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
        Updates the current voice for synthesis and adjusts the language if needed.

        The new voice may imply a different language based on its prefix.
        The method updates the current voice and, if a valid language code is found,
        updates the language accordingly.

        Args:
            voice_name (str): The new voice identifier (e.g., "af_heart").
        """
        self.current_voice_name = voice_name
        # Determine if the new voice suggests a different language.
        lang = self._get_lang_code_from_voice(voice_name)
        if lang:
            self.current_lang = lang
        if self.debug:
            print(f"[KokoroEngine] Voice set to: {voice_name} (lang: {self.current_lang})")


    def set_speed(self, speed: float):
        """
        Sets the speed for speech synthesis.

        Args:
            speed (float): The speed factor (e.g., 1.0 for normal speed,
                           values >1.0 for faster, and values <1.0 for slower).
        """
        self.speed = speed
        if self.debug:
            print(f"[KokoroEngine] Speed set to: {self.speed}")

    def get_voices(self) -> List[str]:
        """
        Retrieves a list of all supported voice identifiers, categorized by language and gender.

        The voices are grouped by language, and within each language, they are
        distinguished as female and male voices.

        Returns:
            List[str]: A list containing all available voice names.
        """
        return [
            # American English (lang_code='a')
            # Female voices (11)
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            # Male voices (9)
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
            "am_michael", "am_onyx", "am_puck", "am_santa",

            # British English (lang_code='b')
            # Female voices (4)
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            # Male voices (4)
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",

            # Japanese (lang_code='j')
            # Female voices (4)
            "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
            # Male voices (1)
            "jm_kumo",

            # Mandarin Chinese (lang_code='z')
            # Female voices (4)
            "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
            # Male voices (4)
            "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",

            # Spanish (lang_code='e')
            # Female voices (1)
            "ef_dora",
            # Male voices (2)
            "em_alex", "em_santa",

            # French (lang_code='f')
            # Female voices (1)
            "ff_siwis",

            # Hindi (lang_code='h')
            # Female voices (2)
            "hf_alpha", "hf_beta",
            # Male voices (2)
            "hm_omega", "hm_psi",

            # Italian (lang_code='i')
            # Female voices (1)
            "if_sara",
            # Male voices (1)
            "im_nicola",

            # Brazilian Portuguese (lang_code='p')
            # Female voices (1)
            "pf_dora",
            # Male voices (2)
            "pm_alex", "pm_santa",
        ]

    def shutdown(self):
        """
        Shuts down the KokoroEngine and performs cleanup if necessary.

        Currently, the KPipeline does not require special cleanup procedures.
        This method is provided for compatibility and future extensibility.
        """
        if self.debug:
            print("[KokoroEngine] Shutdown called.")
        # No additional cleanup required for KPipeline.
        pass
