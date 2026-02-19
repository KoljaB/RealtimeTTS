"""
ModelsLab TTS Engine for RealtimeTTS

A text-to-speech engine that uses ModelsLab's TTS API.
ModelsLab offers high-quality TTS with support for multiple languages and voices.

Installation:
    pip install requests

Usage:
    from RealtimeTTS import TextToAudioStream
    from RealtimeTTS.modelslab_engine import ModelsLabEngine

    engine = ModelsLabEngine(
        api_key="YOUR_MODELSLAB_API_KEY",
        voice="male1",  # or any available voice
        language="english"
    )
    
    stream = TextToAudioStream(engine)
    stream.play("Hello, this is a test of the ModelsLab TTS engine.")

Available voices: male1, male2, male3, female1, female2, female3
Available languages: english, german, french, spanish, italian, portuguese, polish, russian, dutch, arabic, chinese, japanese, korean, hindi
"""

import io
import json
import time
import logging
import requests
from typing import Optional, Dict, Any, List

from RealtimeTTS import BaseEngine
from RealtimeTTS.audio_engine import AudioEngine

logger = logging.getLogger(__name__)


class ModelsLabEngine(BaseEngine):
    """
    ModelsLab TTS Engine for RealtimeTTS.
    
    Provides text-to-speech synthesis using the ModelsLab API.
    Supports multiple voices and languages with low latency.
    """
    
    # Default API endpoint
    BASE_URL = "https://modelslab.com/api/v6/voice"
    
    # Default configuration
    DEFAULT_VOICE = "male1"
    DEFAULT_LANGUAGE = "english"
    DEFAULT_SPEED = 1.0
    
    # Available voices (these are example voices - check ModelsLab docs for current list)
    VOICES = [
        "male1", "male2", "male3",
        "female1", "female2", "female3"
    ]
    
    # Supported languages
    LANGUAGES = [
        "english", "german", "french", "spanish", "italic",
        "portuguese", "polish", "russian", "dutch", "arabic",
        "chinese", "japanese", "korean", "hindi"
    ]
    
    def __init__(
        self,
        api_key: str,
        voice: str = DEFAULT_VOICE,
        language: str = DEFAULT_LANGUAGE,
        speed: float = DEFAULT_SPEED,
        output_format: str = "mp3",
        **kwargs
    ):
        """
        Initialize the ModelsLab TTS Engine.
        
        Args:
            api_key: Your ModelsLab API key
            voice: Voice ID to use (default: male1)
            language: Language for TTS (default: english)
            speed: Speech speed (default: 1.0)
            output_format: Output format (default: mp3)
            **kwargs: Additional parameters
        """
        super().__init__()
        self.api_key = api_key
        self.voice = voice
        self.language = language
        self.speed = speed
        self.output_format = output_format
        self.extra_params = kwargs
        
        # Validate voice
        if voice not in self.VOICES:
            logger.warning(
                f"Voice '{voice}' not in known voices. "
                f"Available: {', '.join(self.VOICES)}. Using anyway."
            )
        
        # Validate language
        if language not in self.LANGUAGES:
            logger.warning(
                f"Language '{language}' not in known languages. "
                f"Available: {', '.join(self.LANGUAGES)}. Using anyway."
            )
        
        self._audio_engine: Optional[AudioEngine] = None
        self._stream = None
        
        logger.info(
            f"Initialized ModelsLab engine with voice={voice}, "
            f"language={language}, speed={speed}"
        )
    
    @property
    def voice(self) -> str:
        return self._voice
    
    @voice.setter
    def voice(self, value: str):
        self._voice = value
    
    @property
    def language(self) -> str:
        return self._language
    
    @language.setter
    def language(self, value: str):
        self._language = value
    
    @property
    def speed(self) -> float:
        return self._speed
    
    @speed.setter
    def speed(self, value: float):
        if value <= 0:
            raise ValueError("Speed must be positive")
        self._speed = value
    
    def _synthesize(self, text: str) -> bytes:
        """
        Synthesize speech from text using ModelsLab API.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as bytes
        """
        payload = {
            "key": self.api_key,
            "prompt": text,
            "voice_id": self.voice,
            "language": self.language,
            "speed": self.speed,
            "emotion": self.extra_params.get("emotion", False),
            "temp": self.extra_params.get("temp", False),
        }
        
        # Add optional parameters
        if "webhook" in self.extra_params:
            payload["webhook"] = self.extra_params["webhook"]
        if "track_id" in self.extra_params:
            payload["track_id"] = self.extra_params["track_id"]
        
        try:
            response = requests.post(
                f"{self.BASE_URL}/text_to_speech",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Handle async response (if status is processing)
            if result.get("status") == "processing":
                request_id = result.get("id")
                if request_id:
                    # Poll for result
                    max_retries = 30
                    for _ in range(max_retries):
                        time.sleep(1)
                        fetch_response = requests.post(
                            f"{self.BASE_URL}/fetch/{request_id}",
                            json={"key": self.api_key},
                            timeout=30
                        )
                        fetch_result = fetch_response.json()
                        if fetch_result.get("status") == "success":
                            result = fetch_result
                            break
                        elif fetch_result.get("status") == "error":
                            raise Exception(fetch_result.get("message", "TTS generation failed"))
            
            if result.get("status") == "success":
                # Get audio URL from output
                audio_url = result.get("output", [None])[0]
                if audio_url:
                    # Download audio file
                    audio_response = requests.get(audio_url, timeout=30)
                    audio_response.raise_for_status()
                    return audio_response.content
                else:
                    raise Exception("No audio URL in response")
            else:
                raise Exception(result.get("message", "TTS generation failed"))
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"ModelsLab API request failed: {str(e)}")
    
    def synthesize(self, text: str) -> bytes:
        """
        Synthesize speech from text.
        
        This is a blocking call that returns the full audio.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as bytes
        """
        return self._synthesize(text)
    
    def synthesize_streaming(self, text: str, callback: callable):
        """
        Synthesize speech in a streaming fashion.
        
        For ModelsLab, we synthesize the full text and stream in chunks.
        
        Args:
            text: Text to synthesize
            callback: Function to call with audio chunks
        """
        audio_data = self._synthesize(text)
        
        # For MP3, we can't easily split, so send in chunks
        # Use a chunk size that works for streaming
        chunk_size = 4096
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            callback(chunk)
    
    def play(self, text_generator, output_device_index: Optional[int] = None, **kwargs):
        """
        Play text as speech.
        
        Args:
            text_generator: Generator yielding text to speak
            output_device_index: Audio output device index
            **kwargs: Additional playback parameters
        """
        # Get audio data from generator
        text = ""
        for text_chunk in text_generator:
            text += text_chunk
        
        if not text.strip():
            return
        
        # Synthesize
        audio_data = self._synthesize(text)
        
        # Play using audio engine
        if self._audio_engine is None:
            self._audio_engine = AudioEngine()
        
        # Play the audio
        self._audio_engine.play(
            io.BytesIO(audio_data),
            output_device_index=output_device_index
        )
    
    def play_async(self, text_generator, output_device_index: Optional[int] = None, **kwargs):
        """
        Play text as speech asynchronously.
        
        Args:
            text_generator: Generator yielding text to speak
            output_device_index: Audio output device index
            **kwargs: Additional playback parameters
        """
        import threading
        
        thread = threading.Thread(
            target=self.play,
            args=(text_generator, output_device_index),
            kwargs=kwargs
        )
        thread.daemon = True
        thread.start()
    
    def stop(self):
        """Stop playback."""
        if self._audio_engine:
            self._audio_engine.stop()
    
    def is_available(self) -> bool:
        """
        Check if the engine is available (API key configured).
        
        Returns:
            True if API key is set
        """
        return bool(self.api_key and self.api_key != "YOUR_API_KEY")
    
    def get_voices(self) -> List[str]:
        """Get list of available voices."""
        return self.VOICES.copy()
    
    def get_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.LANGUAGES.copy()
    
    def __repr__(self) -> str:
        return (
            f"ModelsLabEngine(voice='{self.voice}', "
            f"language='{self.language}', speed={self.speed})"
        )


# Convenience function to create engine from environment
def create_engine_from_env():
    """
    Create a ModelsLabEngine from environment variables.
    
    Environment variables:
        MODELSLAB_API_KEY: Your ModelsLab API key
        MODELSLAB_VOICE: Voice to use (optional)
        MODELSLAB_LANGUAGE: Language to use (optional)
        MODELSLAB_SPEED: Speech speed (optional)
    
    Returns:
        ModelsLabEngine instance or None if no API key
    """
    import os
    
    api_key = os.environ.get("MODELSLAB_API_KEY")
    if not api_key:
        return None
    
    return ModelsLabEngine(
        api_key=api_key,
        voice=os.environ.get("MODELSLAB_VOICE", ModelsLabEngine.DEFAULT_VOICE),
        language=os.environ.get("MODELSLAB_LANGUAGE", ModelsLabEngine.DEFAULT_LANGUAGE),
        speed=float(os.environ.get("MODELSLAB_SPEED", 1.0))
    )


# Example usage
if __name__ == "__main__":
    import os
    
    # Set API key (or use environment variable)
    API_KEY = os.environ.get("MODELSLAB_API_KEY", "YOUR_API_KEY")
    
    if API_KEY == "YOUR_API_KEY":
        print("Please set MODELSLAB_API_KEY environment variable or update the script")
        print(__doc__)
    else:
        # Create engine
        engine = ModelsLabEngine(
            api_key=API_KEY,
            voice="male1",
            language="english"
        )
        
        # Check availability
        print(f"Engine available: {engine.is_available()}")
        print(f"Voices: {engine.get_voices()}")
        print(f"Languages: {engine.get_languages()}")
        
        # Synthesize sample
        print("\nSynthesizing: 'Hello, this is a test of the ModelsLab TTS engine.'")
        audio = engine.synthesize("Hello, this is a test of the ModelsLab TTS engine.")
        print(f"Generated {len(audio)} bytes of audio")
        
        # Optionally save to file
        with open("test_output.mp3", "wb") as f:
            f.write(audio)
        print("Saved to test_output.mp3")
