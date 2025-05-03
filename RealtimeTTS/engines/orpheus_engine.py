import json
import time
import logging
import pyaudio
import requests
import traceback
import numpy as np
from queue import Queue
from typing import Optional, Union
from .base_engine import BaseEngine

# Default configuration values
DEFAULT_API_URL = "http://127.0.0.1:1234/v1/completions"
DEFAULT_HEADERS = {"Content-Type": "application/json"}
DEFAULT_MODEL = "orpheus-3b-0.1-ft"
DEFAULT_VOICE = "tara"
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
SAMPLE_RATE = 24000  # Specific sample rate for Orpheus

# Special token definitions for prompt formatting and token decoding
START_TOKEN_ID = 128259
END_TOKEN_IDS = [128009, 128260, 128261, 128257]
CUSTOM_TOKEN_PREFIX = "<custom_token_"


class OrpheusVoice:
    """
    Represents the configuration for an Orpheus voice.
    
    Attributes:
        name (str): The name of the voice. Must be one of the AVAILABLE_VOICES.
    
    Raises:
        ValueError: If the voice name provided is not in AVAILABLE_VOICES.
    """
    def __init__(self, name: str):
        # if name not in AVAILABLE_VOICES:
        #     raise ValueError(f"Invalid voice '{name}'. Available voices: {AVAILABLE_VOICES}")
        self.name = name

    def __repr__(self):
        return f"OrpheusVoice(name='{self.name}')"


class OrpheusEngine(BaseEngine):
    """
    Real-time Text-to-Speech (TTS) engine for the Orpheus model via LM Studio API.
    
    This engine supports real-time token generation, audio synthesis, and voice configuration.
    """
    
    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        model: str = DEFAULT_MODEL,
        headers: dict = DEFAULT_HEADERS,
        voice: Optional[OrpheusVoice] = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 1200,
        repetition_penalty: float = 1.1,
        debug: bool = False
    ):
        """
        Initialize the Orpheus TTS engine with the given parameters.

        Args:
            api_url (str): Endpoint URL for the LM Studio API.
            model (str): Model name to use for synthesis.
            headers (dict): HTTP headers for API requests.
            voice (Optional[OrpheusVoice]): OrpheusVoice configuration. Defaults to DEFAULT_VOICE.
            temperature (float): Sampling temperature (0-1) for text generation.
            top_p (float): Top-p sampling parameter for controlling diversity.
            max_tokens (int): Maximum tokens to generate per API request.
            repetition_penalty (float): Penalty factor for repeated phrases.
            debug (bool): Flag to enable debug output.
        """
        super().__init__()
        self.api_url = api_url
        self.model = model
        self.headers = headers
        self.voice = voice or OrpheusVoice(DEFAULT_VOICE)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.debug = debug
        self.queue = Queue()
        self.post_init()

    def post_init(self):
        """Set up additional engine attributes."""
        self.engine_name = "orpheus"

    def get_stream_info(self):
        """
        Retrieve PyAudio stream configuration.

        Returns:
            tuple: Format, channel count, and sample rate for PyAudio.
        """
        return pyaudio.paInt16, 1, SAMPLE_RATE

    def synthesize(self, text: str) -> bool:
        """
        Convert text to speech and stream audio data.

        Args:
            text (str): The input text to be synthesized.

        Returns:
            bool: True if synthesis was successful, False otherwise.
        """
        super().synthesize(text)

        try:
            # Process tokens and put generated audio chunks into the queue
            for audio_chunk in self._token_decoder(self._generate_tokens(text)):
                # bail out immediately if someone called .stop()
                if self.stop_synthesis_event.is_set():
                    logging.info("OrpheusEngine: synthesis stopped by user")
                    return False
                print(f"Audio chunk size: {len(audio_chunk)}")
                self.queue.put(audio_chunk)
            return True
        except Exception as e:
            traceback.print_exc()
            logging.error(f"Synthesis error: {e}")
            return False

    def synthesize(self, text: str) -> bool:
        """
        Convert text to speech and stream audio data via Orpheus.
        Drops initial and trailing near-silent chunks.
        """
        super().synthesize(text)

        try:
            for audio_chunk in self._token_decoder(self._generate_tokens(text)):
                # bail out if user called .stop()
                if self.stop_synthesis_event.is_set():
                    logging.info("OrpheusEngine: synthesis stopped by user")
                    return False

                # forward this chunk
                self.queue.put(audio_chunk)

            return True

        except Exception as e:
            traceback.print_exc()
            logging.error(f"Synthesis error: {e}")
            return False


    def _generate_tokens(self, prompt: str):
        """
        Generate a token stream using the LM Studio API.

        Args:
            prompt (str): The input text prompt.

        Yields:
            str: Each token's text as it is received from the API.
        """
        logging.debug(f"Generating tokens for prompt: {prompt}")
        formatted_prompt = self._format_prompt(prompt)
        
        payload = {
            "model": self.model,
            "prompt": formatted_prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repetition_penalty,
            "stream": True
        }

        try:
            logging.debug(f"Requesting API URL: {self.api_url} with payload: {payload} and headers: {self.headers}")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                stream=True
            )
            response.raise_for_status()

            token_counter = 0
            start_time = time.time()  # Start timing token generation
            for line in response.iter_lines():
                # stop on demand
                if self.stop_synthesis_event.is_set():
                    logging.debug("OrpheusEngine: token generation aborted")
                    break
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and data['choices']:
                                token_text = data['choices'][0].get('text', '')
                                if token_text:
                                    token_counter += 1
                                    # Print the time it took to get the first token
                                    if token_counter == 1:
                                        elapsed = time.time() - start_time
                                        logging.info(f"Time to first token: {elapsed:.2f} seconds")
                                    yield token_text
                        except json.JSONDecodeError as e:
                            logging.error(f"Error decoding JSON: {e}")
                            continue

        except requests.RequestException as e:
            logging.error(f"API request failed: {e}")

    def _format_prompt(self, prompt: str) -> str:
        """
        Format the text prompt with special tokens required by Orpheus.

        Args:
            prompt (str): The raw text prompt.

        Returns:
            str: The formatted prompt including voice and termination token.
        """
        return f"<|audio|>{self.voice.name}: {prompt}<|eot_id|>"

    def _token_decoder(self, token_gen):
        """
        Decode tokens from the generator and convert them into audio samples.

        This method aggregates tokens in a buffer and converts them into audio chunks
        once enough tokens have been collected.

        Args:
            token_gen: Generator yielding token strings.

        Yields:
            Audio samples ready to be streamed.
        """
        buffer = []
        count = 0

        logging.debug("Starting token decoding from token generator.")
        for token_text in token_gen:
            # bail out if stop was requested
            if self.stop_synthesis_event.is_set():
                logging.debug("OrpheusEngine: token decoding aborted")
                break
            token = self.turn_token_into_id(token_text, count)
            if token is not None and token > 0:
                buffer.append(token)
                count += 1

                # Process every 7 tokens after an initial threshold
                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:]
                    audio_samples = self._convert_buffer(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples

    def turn_token_into_id(self, token_string: str, index: int) -> Optional[int]:
        """
        Convert a token string to a numeric ID for audio processing.

        The conversion takes into account the custom token prefix and an index-based offset.

        Args:
            token_string (str): The token text.
            index (int): The current token index.

        Returns:
            Optional[int]: The numeric token ID or None if conversion fails.
        """
        token_string = token_string.strip()
        last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
        
        if last_token_start == -1:
            return None
        
        last_token = token_string[last_token_start:]
        
        if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
            try:
                number_str = last_token[14:-1]
                token_id = int(number_str) - 10 - ((index % 7) * 4096)
                return token_id
            except ValueError:
                return None
        else:
            return None

    def _convert_buffer(self, multiframe, count: int):
        """
        Convert a buffer of token frames into audio samples.

        This method uses an external decoder to convert the collected token frames.

        Args:
            multiframe: List of token IDs to be converted.
            count (int): The current token count (used for conversion logic).

        Returns:
            Converted audio samples if successful; otherwise, None.
        """
        try:
            from .orpheus_decoder import convert_to_audio as orpheus_convert_to_audio
            converted = orpheus_convert_to_audio(multiframe, count)
            if converted is None:
                logging.warning("Conversion returned None.")
            return converted
        except Exception as e:
            logging.error(f"Failed to convert buffer to audio: {e}")
        logging.info("Returning None after failed conversion.")
        return None

    def get_voices(self):
        """
        Retrieve the list of available voices.

        Returns:
            list: A list of OrpheusVoice instances for each available voice.
        """
        return [OrpheusVoice(name) for name in AVAILABLE_VOICES]

    def set_voice(self, voice: Union[str, OrpheusVoice]):
        """
        Set the current voice for synthesis.

        Args:
            voice (Union[str, OrpheusVoice]): The voice name or an OrpheusVoice instance.

        Raises:
            ValueError: If the provided voice name is invalid.
            TypeError: If the voice argument is neither a string nor an OrpheusVoice instance.
        """
        if isinstance(voice, str):
            # if voice not in AVAILABLE_VOICES:
            #     raise ValueError(f"Invalid voice '{voice}'")
            self.voice = OrpheusVoice(voice)
        elif isinstance(voice, OrpheusVoice):
            self.voice = voice
        else:
            raise TypeError("Voice must be a string or an OrpheusVoice instance.")

    def set_voice_parameters(self, **kwargs):
        """
        Update voice generation parameters.

        Valid parameters include 'temperature', 'top_p', 'max_tokens', and 'repetition_penalty'.

        Args:
            **kwargs: Arbitrary keyword arguments for valid voice parameters.
        """
        valid_params = ['temperature', 'top_p', 'max_tokens', 'repetition_penalty']
        for param, value in kwargs.items():
            if param in valid_params:
                setattr(self, param, value)
            elif self.debug:
                logging.warning(f"Ignoring invalid parameter: {param}")

    def __del__(self):
        """
        Destructor to clean up resources.

        Puts a None into the queue to signal termination of audio processing.
        """
        self.queue.put(None)
