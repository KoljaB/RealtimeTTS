"""
Requires:
- pip install kokoro>=0.3.4 soundfile torch
- For non-English languages, install espeak-ng:
  !apt-get -qq -y install espeak-ng
"""

from .base_engine import BaseEngine, TimingInfo
from queue import Queue
from typing import List, Union
import numpy as np
import traceback
import pyaudio
import time

# Make sure torch is installed
import torch
import re

# Import the text-to-speech pipeline from the Kokoro package.
from kokoro import KPipeline

class KokoroEngine(BaseEngine):
    """
    A text-to-speech (TTS) engine utilizing the Kokoro pipeline, now with support
    for blending multiple voices by parsing "voice formulas."

    Example usage of a voice formula:
        engine.set_voice("0.3*af_sarah + 0.7*am_adam")

    That string will be parsed to load 'af_sarah' and 'am_adam', then combined
    with a weighted average. The final voicepack is cached under the same formula
    key, so you can reuse it without re-computation.
    """

    def __init__(
            self,
            default_lang_code: str = "a",
            default_voice: str = "af_heart",
            default_speed: float = 1.0,
            debug: bool = False):
        """
        Initializes the KokoroEngine with default settings.

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
        self.speed = default_speed

        self.current_voice_name = default_voice
        # Attempt to derive the language code from the voice name.
        lang_from_voice = self._get_lang_code_from_voice(default_voice)
        self.current_lang = lang_from_voice if lang_from_voice else default_lang_code

        # Create and cache the pipeline for the current language.
        self.pipelines[self.current_lang] = KPipeline(
            repo_id = 'hexgrad/Kokoro-82M',
            lang_code=self.current_lang
        )

        # Cache for formula-based blended voices: { formula_str: torch.FloatTensor }
        self.blended_voices = {}

        if self.debug:
            print(f"[KokoroEngine] Initialized with voice: {self.current_voice_name} (lang: {self.current_lang}), speed: {self.speed}")

    def _get_lang_code_from_voice(self, voice_name: str) -> str:
        """
        Determines the language code from the provided voice name.
        If a voice formula is supplied, we just pick the first voice's code.

        Args:
            voice_name (str): The voice identifier (e.g., "af_heart" or "0.3*af_sarah + 0.7*am_adam").

        Returns:
            str: The language code or None if unknown.
        """
        # If there's a weight spec, parse out the first chunk for detecting language:
        if "*" in voice_name:
            # e.g. "0.3*af_sarah + 0.7*am_adam"
            parts = re.split(r"\+|\s+", voice_name)
            # find something that looks like "0.3*af_sarah"
            for part in parts:
                if "*" in part:
                    # get voice token: e.g. "af_sarah"
                    voice_token = part.split("*")[-1].strip()
                    # fallback to normal logic
                    return self._get_lang_code_from_voice(voice_token)
            return None

        # Standard single-voice approach
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
            # Fallback to first-letter guess
            first_char = voice_name[0].lower() if voice_name else ''
            if first_char in "abjzefhip":
                return first_char
            return None

    def _get_pipeline(self, lang_code: str):
        """
        Retrieves the KPipeline for the specified language code.

        Args:
            lang_code (str): The language code for which to get the pipeline.

        Returns:
            KPipeline: The pipeline instance corresponding to the language code.
        """
        if lang_code not in self.pipelines:
            if self.debug:
                print(f"[KokoroEngine] Creating new pipeline for language code: {lang_code}")
            self.pipelines[lang_code] = KPipeline(
                repo_id = 'hexgrad/Kokoro-82M',
                lang_code=lang_code
            )
        return self.pipelines[lang_code]

    def _parse_mixed_voice_formula(self, formula: str, pipeline: KPipeline) -> torch.FloatTensor:
        """
        Parse a formula like "0.3*af_sarah + 0.7*am_adam" to create a weighted blend of voice Tensors.

        Args:
            formula (str): Weighted formula string.
            pipeline (KPipeline): The pipeline for the relevant language.

        Returns:
            torch.FloatTensor: The blended voice Tensor.
        """
        # If we already have a cached blend, return it
        if formula in self.blended_voices:
            if self.debug:
                print(f"[KokoroEngine] Using cached blended voice for formula: {formula}")
            return self.blended_voices[formula]

        # Otherwise, parse and create a new blend
        # Example formula: "0.3*af_sarah + 0.7*am_adam"
        # Split on "+"
        segments = [seg.strip() for seg in formula.split("+")]
        total_weight = 0.0
        sum_tensor = None

        for seg in segments:
            seg = seg.strip()
            if "*" not in seg:
                raise ValueError(f"Malformed voice formula segment (missing '*'): '{seg}'")

            # e.g. "0.3*af_sarah"
            weight_str, voice_name = seg.split("*", 1)
            weight = float(weight_str.strip())
            voice_name = voice_name.strip()
            total_weight += weight

            # Load single voice or cached voice
            voice_tensor = pipeline.load_single_voice(voice_name)

            # Weighted sum
            fraction = torch.tensor(weight)  # if you want float conversion
            if sum_tensor is None:
                sum_tensor = fraction * voice_tensor
            else:
                sum_tensor += fraction * voice_tensor

        if total_weight == 0:
            raise ValueError(f"Total voice weight is zero in formula: {formula}")

        # Normalize by total_weight
        sum_tensor /= total_weight

        # Cache the final
        self.blended_voices[formula] = sum_tensor
        return sum_tensor

    def get_stream_info(self):
        """
        Provides the PyAudio stream configuration for the synthesized audio.

        Returns:
            tuple: (pyaudio.paInt16, 1, 24000)
        """
        # Kokoro uses 24 kHz sampling rate, mono channel, with 16-bit samples.
        return (pyaudio.paInt16, 1, 24000)

    def synthesize(self, text: str) -> bool:
        """
        Converts the input text into speech audio, in chunks, placing the data into self.queue.

        Args:
            text (str): The text string to synthesize.

        Returns:
            bool: True if synthesis is successful, False otherwise.
        """
        start_time = time.time()
        try:
            if self.debug:
                print(f"[KokoroEngine] Synthesizing with language code: {self.current_lang} and speed: {self.speed}")

            # Pull the pipeline for the current language
            pipeline = self._get_pipeline(self.current_lang)

        except Exception as e:
            traceback.print_exc()
            print(f"[KokoroEngine] Error creating kokoro KPipeline: {e}")
            return False

        try:
            # If current_voice_name is a formula, parse it. Otherwise, use it as is.
            voice_arg: Union[str, torch.FloatTensor] = self.current_voice_name
            if "*" in self.current_voice_name:  # basic check for formula
                voice_arg = self._parse_mixed_voice_formula(self.current_voice_name, pipeline)

        except Exception as e:
            traceback.print_exc()
            print(f"[KokoroEngine] Error creating / mixing kokoro voice: {e}")
            return False

        try:
            # Generate audio in chunks from the pipeline
            generator = pipeline(text, voice=voice_arg, speed=self.speed)

        except Exception as e:
            traceback.print_exc()
            print(f"[KokoroEngine] Error creating generator from kokoro pipeline for text: {e}")
            return False

        try:
            if generator and generator is not None:
                for index, result in enumerate(generator):
                    graphemes = result.graphemes # str
                    phonemes = result.phonemes # str
                    audio_float32 = result.audio.cpu().numpy()
                    tokens = result.tokens # List[en.MToken]

                    if self.debug:
                        if graphemes:
                            print(f"Graphemes available for chunk {index}: {graphemes}")
                        else:
                            print(f"No graphemes available for chunk {index}")
                        if phonemes:
                            print(f"Phonemes available for chunk {index}: {phonemes}")
                        else:
                            print(f"No phonemes available for chunk {index}")

                    if tokens:
                        if self.debug:
                            print(f"Timing tokens available for chunk {index}:")
                        for t in tokens:
                            timingInfo = TimingInfo(
                                t.start_ts + self.audio_duration,
                                t.end_ts + self.audio_duration,
                                t.text
                            )
                            self.timings.put(timingInfo)
                            if self.debug:
                                print(f"Token: {t.text} ({t.start_ts:.2f}s - {t.end_ts:.2f}s)")
                    else:
                        if self.debug:
                            print(f"No timing tokens available for chunk {index}")

                    audio_int16 = (audio_float32 * 32767).astype(np.int16).tobytes()
                    audio_length_in_seconds = len(audio_float32) / 24000
                    self.audio_duration += audio_length_in_seconds
                    self.queue.put(audio_int16)


                if self.debug:
                    duration = time.time() - start_time
                    print(f"[KokoroEngine] Synthesis completed in {duration:.3f}s.")

                return True

            else:
                print(f"[KokoroEngine] No generator created for text: {text}")

        except Exception as e:
            traceback.print_exc()
            print(f"[KokoroEngine] Error generating audio: {e}")
            return False

    def set_voice(self, voice_name: str):
        """
        Updates the current voice or voice formula. If it's a single voice (e.g. "af_heart"),
        we detect language as usual. If it contains '*' or '+', we treat it as a blend formula.

        Args:
            voice_name (str): The new voice identifier (e.g., "af_heart" or "0.3*af_sarah + 0.7*am_adam").
        """
        self.current_voice_name = voice_name
        # Attempt to detect language from the first relevant voice chunk
        lang = self._get_lang_code_from_voice(voice_name)
        if lang:
            self.current_lang = lang

        if self.debug:
            print(f"[KokoroEngine] Voice set to: {voice_name} (lang: {self.current_lang})")

    def set_speed(self, speed: float):
        """
        Sets the speed for speech synthesis.

        Args:
            speed (float): The speed factor (e.g., 1.0 for normal speed).
        """
        self.speed = speed
        if self.debug:
            print(f"[KokoroEngine] Speed set to: {self.speed}")

    def get_voices(self) -> List[str]:
        """
        Retrieves a list of all supported voice identifiers.

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
        """
        if self.debug:
            print("[KokoroEngine] Shutdown called.")
        # No additional cleanup required for KPipeline.
        pass

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets the voice parameters to be used for speech synthesis.

        Args:
            **voice_parameters: The voice parameters to be used for speech synthesis.
        """
        if "speed" in voice_parameters:
            self.set_speed(voice_parameters["speed"])
