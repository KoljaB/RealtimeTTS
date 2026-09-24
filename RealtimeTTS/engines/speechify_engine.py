from .base_engine import BaseEngine
from typing import Union
import requests
import pyaudio
import time
import os
import logging


class SpeechifyVoice:
    def __init__(self, voice_id, name="", locale="", gender=""):
        self.voice_id = voice_id
        self.name = name
        self.locale = locale
        self.gender = gender

    def __repr__(self):
        return f"{self.name} ({self.voice_id})"


class SpeechifyEngine(BaseEngine):
    """
    Speechify text-to-speech engine for RealtimeTTS.

    Streams raw 16-bit PCM from the Speechify API (POST /v1/audio/stream),
    so playback starts while the rest of the sentence is still generating.

    API Docs: https://docs.speechify.ai
    """

    SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)

    def __init__(
        self,
        api_key: str = None,
        voice_id: str = "geffen_32",
        model: str = "simba-3.2",
        language: str = None,
        sample_rate: int = 24000,
        loudness_normalization: bool = None,
        text_normalization: bool = None,
        base_url: str = "https://api.speechify.ai",
        chunk_size: int = 4096,
        debug: bool = False,
    ):
        """
        Initializes a Speechify realtime text to speech engine object.

        Args:
            api_key (str, optional): Speechify API key. Defaults to SPEECHIFY_API_KEY env var.
            voice_id (str, optional): Voice id, see get_voices(). Defaults to "geffen_32".
            model (str, optional): "simba-3.2" (English) or "simba-3.0" (multilingual).
                Defaults to "simba-3.2".
            language (str, optional): Language of the input, e.g. "en-US" or "fr-FR".
                If None, the voice's own locale is used.
            sample_rate (int, optional): PCM sample rate. One of 8000, 16000, 22050,
                24000, 44100, 48000. Defaults to 24000.
            loudness_normalization (bool, optional): Normalize output loudness. Adds latency.
            text_normalization (bool, optional): Spell out numbers, dates etc. Adds latency.
            base_url (str, optional): API base URL.
            chunk_size (int, optional): Size in bytes of the audio chunks put on the queue.
            debug (bool, optional): If True, prints debugging information.
        """
        self.api_key = api_key or os.environ.get("SPEECHIFY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Speechify API key is required. Provide it via api_key parameter "
                "or SPEECHIFY_API_KEY environment variable."
            )

        if sample_rate not in self.SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {self.SAMPLE_RATES}")

        self.voice_id = voice_id
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.loudness_normalization = loudness_normalization
        self.text_normalization = text_normalization
        self.base_url = base_url.rstrip("/")
        self.chunk_size = chunk_size
        self.debug = debug
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Speechify-Caller": "realtimetts",
        })

    def post_init(self):
        self.engine_name = "speechify"

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration for Speechify audio
        (raw 16-bit mono PCM at the configured sample rate).

        Returns:
            tuple: (format, channels, sample_rate)
        """
        return pyaudio.paInt16, 1, self.sample_rate

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
            sentence_count (int): The count of sentences synthesized so far, used for tracking progress.

        Returns:
            bool: True if successful, False otherwise.
        """
        super().synthesize(text, sentence_count)

        if self.debug:
            print(f"Speechify synthesizing: \"{text}\"")

        payload = {
            "input": text,
            "voice_id": self.voice_id,
            "model": self.model,
            "output_format": f"pcm_{self.sample_rate}",
        }
        if self.language:
            payload["language"] = self.language

        options = {}
        if self.loudness_normalization is not None:
            options["loudness_normalization"] = self.loudness_normalization
        if self.text_normalization is not None:
            options["text_normalization"] = self.text_normalization
        if options:
            payload["options"] = options

        start_time = time.time()
        first_chunk = True
        leftover = b""

        try:
            with self.session.post(
                f"{self.base_url}/v1/audio/stream",
                json=payload,
                stream=True,
                timeout=60,
            ) as response:
                if response.status_code != 200:
                    logging.error(
                        f"Speechify TTS API error {response.status_code}: {response.text}"
                    )
                    return False

                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if self.stop_synthesis_event.is_set():
                        return False
                    if not chunk:
                        continue

                    if first_chunk:
                        first_chunk = False
                        if self.debug:
                            print(f"Speechify time to first chunk: {time.time() - start_time:.2f} seconds")

                    # Keep whole 16-bit samples in every chunk
                    chunk = leftover + chunk
                    cut = len(chunk) - (len(chunk) % 2)
                    leftover = chunk[cut:]
                    if cut:
                        self.queue.put(chunk[:cut])

            return True

        except requests.exceptions.RequestException as e:
            logging.error(f"Speechify TTS request error: {e}")
            return False

    def get_voices(self):
        """
        Retrieves the voices available to this API key.

        Returns:
            list: List of SpeechifyVoice objects.
        """
        try:
            response = self.session.get(f"{self.base_url}/v1/voices", timeout=30)
            response.raise_for_status()
            return [
                SpeechifyVoice(
                    voice_id=v.get("id"),
                    name=v.get("display_name", ""),
                    locale=v.get("locale", ""),
                    gender=v.get("gender", ""),
                )
                for v in response.json()
            ]
        except requests.exceptions.RequestException as e:
            logging.error(f"Speechify voices request error: {e}")
            return []

    def set_voice(self, voice: Union[str, SpeechifyVoice]):
        """
        Sets the voice to be used for speech synthesis.

        Args:
            voice (Union[str, SpeechifyVoice]): Voice id or SpeechifyVoice object.
        """
        if isinstance(voice, SpeechifyVoice):
            self.voice_id = voice.voice_id
        else:
            self.voice_id = voice

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets voice parameters for synthesis.

        Supported keys: model, language, loudness_normalization, text_normalization
        """
        for key in ("model", "language", "loudness_normalization", "text_normalization"):
            if key in voice_parameters:
                setattr(self, key, voice_parameters[key])

    def shutdown(self):
        """
        Shuts down the engine and releases resources.
        """
        self.session.close()
