from .base_engine import BaseEngine
from .elevenlabs_engine import ElevenlabsEngine
import requests
import io
import pyaudio
import time
import os

# ANSI escape codes for colors
COLOR_GREEN = "\033[92m"   # Green for text messages
COLOR_YELLOW = "\033[93m"  # Yellow for time messages
COLOR_WHITE = "\033[97m"   # White for all else
COLOR_RESET = "\033[0m"


class ModelsLabVoice:
    def __init__(self, name, voice_id, language="american english"):
        self.name = name
        self.voice_id = voice_id
        self.language = language

    def __repr__(self):
        return f"{self.name} ({self.voice_id})"


class ModelsLabEngine(BaseEngine):
    """
    ModelsLab text-to-speech engine for RealtimeTTS.
    
    Uses the ModelsLab API to convert text to speech with support for
    multiple languages and voice presets.
    
    API Docs: https://docs.modelslab.com/voice-cloning/text-to-speech
    """
    
    # Pre-trained voice IDs available from ModelsLab
    VOICES = [
        # Female voices
        {"name": "madison", "voice_id": "madison", "language": "american english"},
        {"name": "jennifer", "voice_id": "jennifer", "language": "american english"},
        {"name": "mia", "voice_id": "mia", "language": "american english"},
        {"name": "leigha", "voice_id": "leigha", "language": "american english"},
        {"name": "tara", "voice_id": "tara", "language": "american english"},
        {"name": "leah", "voice_id": "leah", "language": "american english"},
        {"name": "jess", "voice_id": "jess", "language": "american english"},
        {"name": "zoe", "voice_id": "zoe", "language": "american english"},
        # Male voices
        {"name": "james", "voice_id": "james", "language": "american english"},
        {"name": "daniel", "voice_id": "daniel", "language": "american english"},
        {"name": "richard", "voice_id": "richard", "language": "american english"},
        {"name": "leo", "voice_id": "leo", "language": "american english"},
        {"name": "dan", "voice_id": "dan", "language": "american english"},
        {"name": "zac", "voice_id": "zac", "language": "american english"},
        # British English
        {"name": "george", "voice_id": "george", "language": "british english"},
        {"name": "emma", "voice_id": "emma", "language": "british english"},
        # Spanish
        {"name": "martin", "voice_id": "martin", "language": "spanish"},
        {"name": "maria", "voice_id": "maria", "language": "spanish"},
        # French
        {"name": "henri", "voice_id": "henri", "language": "french"},
        {"name": "claire", "voice_id": "claire", "language": "french"},
        # German
        {"name": "klaus", "voice_id": "klaus", "language": "german"},
        {"name": "lena", "voice_id": "lena", "language": "german"},
        # Italian
        {"name": "giorgio", "voice_id": "giorgio", "language": "italian"},
        {"name": "giulia", "voice_id": "giulia", "language": "italian"},
        # Japanese
        {"name": "daichi", "voice_id": "daichi", "language": "japanese"},
        {"name": "haruka", "voice_id": "haruka", "language": "japanese"},
        # Hindi
        {"name": "amit", "voice_id": "amit", "language": "hindi"},
        {"name": "priya", "voice_id": "priya", "language": "hindi"},
        # Mandarin Chinese
        {"name": "wang", "voice_id": "wang", "language": "mandarin chinese"},
        {"name": "mei", "voice_id": "mei", "language": "mandarin chinese"},
        # Brazilian Portuguese
        {"name": "antonio", "voice_id": "antonio", "language": "brazilian portuguese"},
        {"name": "fernanda", "voice_id": "fernanda", "language": "brazilian portuguese"},
    ]

    def __init__(
        self,
        api_key: str = None,
        voice: str = "madison",
        language: str = "american english",
        speed: float = 1.0,
        emotion: bool = False,
        debug: bool = False,
    ):
        """
        Initialize the ModelsLab TTS engine.

        Args:
            api_key (str, optional): ModelsLab API key. Defaults to MODELSLAB_API_KEY env var.
            voice (str, optional): Voice ID to use. Defaults to "madison".
            language (str, optional): Language for speech synthesis.
                Options: american english, british english, spanish, japanese, 
                mandarin chinese, french, brazilian portuguese, hindi, italian
                Defaults to "american english".
            speed (float, optional): Speech speed. Defaults to 1.0.
            emotion (bool, optional): Enable emotion support (English only). Defaults to False.
            debug (bool, optional): Print debug information. Defaults to False.
        """
        self.api_key = api_key or os.environ.get("MODELSLAB_API_KEY")
        if not self.api_key:
            raise ValueError("ModelsLab API key is required. Provide it via api_key parameter or MODELSLAB_API_KEY env var.")
        
        self.voice = voice
        self.language = language
        self.speed = speed
        self.emotion = emotion
        self.debug = debug
        
        self.queue = None
        self.base_url = "https://modelslab.com/api/v6/voice/text_to_speech"

    def post_init(self):
        self.engine_name = "modelslab"

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration for ModelsLab audio.
        ModelsLab returns MP3 audio at 22050 Hz sample rate.

        Returns:
            tuple: (format, channels, sample_rate)
        """
        return pyaudio.paCustomFormat, 1, 22050

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.

        Returns:
            bool: True if synthesis succeeded.
        """
        if self.debug:
            print(f"{COLOR_GREEN}ModelsLab synthesizing: {text[:50]}{'...' if len(text) > 50 else ''}{COLOR_RESET}")

        start_time = time.time()
        first_token_printed = False

        # Build request payload
        payload = {
            "key": self.api_key,
            "prompt": text,
            "voice_id": self.voice,
            "language": self.language,
            "speed": self.speed,
            "emotion": self.emotion,
        }

        try:
            response = requests.post(self.base_url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            if self.debug:
                print(f"{COLOR_WHITE}ModelsLab response: {result.get('status')}{COLOR_RESET}")

            # Check if result is processing (async)
            if result.get("status") == "processing":
                fetch_url = result.get("fetch_result")
                if not fetch_url:
                    if self.debug:
                        print(f"{COLOR_YELLOW}No fetch_url in processing response{COLOR_RESET}")
                    return False
                
                # Poll for result
                max_retries = 30
                for _ in range(max_retries):
                    time.sleep(1)
                    poll_response = requests.get(fetch_url, timeout=30)
                    poll_result = poll_response.json()
                    
                    if poll_result.get("status") == "success":
                        result = poll_result
                        break
                    elif poll_result.get("status") == "error":
                        if self.debug:
                            print(f"{COLOR_YELLOW}ModelsLab error: {poll_result.get('message')}{COLOR_RESET}")
                        return False

            if result.get("status") != "success":
                if self.debug:
                    print(f"{COLOR_YELLOW}ModelsLab synthesis failed: {result.get('message')}{COLOR_RESET}")
                return False

            # Get audio URL from result
            audio_urls = result.get("output") or result.get("proxy_links") or result.get("links")
            if not audio_urls or not audio_urls[0]:
                if self.debug:
                    print(f"{COLOR_YELLOW}No audio URL in response{COLOR_RESET}")
                return False

            audio_url = audio_urls[0]

            # Download and stream audio
            audio_response = requests.get(audio_url, timeout=30)
            audio_response.raise_for_status()

            audio_data = audio_response.content

            if not first_token_printed:
                elapsed = time.time() - start_time
                if self.debug:
                    print(f"{COLOR_YELLOW}ModelsLab time to first audio: {elapsed:.2f} seconds{COLOR_RESET}")
                first_token_printed = True

            # Put audio data in queue (MP3 bytes)
            self.queue.put(audio_data)

            return True

        except requests.exceptions.RequestException as e:
            if self.debug:
                print(f"{COLOR_YELLOW}ModelsLab request error: {e}{COLOR_RESET}")
            return False
        except Exception as e:
            if self.debug:
                print(f"{COLOR_YELLOW}ModelsLab synthesis error: {e}{COLOR_RESET}")
            return False

    def get_voices(self):
        """
        Retrieves the available voices for ModelsLab TTS.

        Returns:
            list: List of ModelsLabVoice objects.
        """
        voice_objects = []
        for voice_data in self.VOICES:
            voice_objects.append(ModelsLabVoice(
                name=voice_data["name"],
                voice_id=voice_data["voice_id"],
                language=voice_data["language"]
            ))
        return voice_objects

    def set_voice(self, voice: str):
        """
        Sets the voice to be used for speech synthesis.

        Args:
            voice (str): Voice ID or name to set.
        """
        # Check if it's a voice ID
        for v in self.VOICES:
            if voice == v["voice_id"]:
                self.voice = v["voice_id"]
                self.language = v["language"]
                return
        
        # Check if it's a voice name
        for v in self.VOICES:
            if voice.lower() == v["name"].lower():
                self.voice = v["voice_id"]
                self.language = v["language"]
                return
        
        # Default to provided voice with current language
        self.voice = voice

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets voice parameters for synthesis.

        Args:
            **voice_parameters: Parameters like speed, language, emotion.
        """
        if "speed" in voice_parameters:
            self.speed = voice_parameters["speed"]
        if "language" in voice_parameters:
            self.language = voice_parameters["language"]
        if "emotion" in voice_parameters:
            self.emotion = voice_parameters["emotion"]
