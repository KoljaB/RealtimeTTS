from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
from .base_engine import BaseEngine
from threading import Thread
from typing import Union
import pyaudio
import torch
import time


class ParlerVoice:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def __repr__(self):
        return self.name

class ParlerEngine(BaseEngine):
    def __init__(
        self,
        voice_prompt="Laura's voice is bright, joyful, and full of energy. Each word is engaging, radiating warmth and positivity that lifts the listener's mood.",
        model_name="parler-tts/parler-tts-mini-v1", # ylacombe/parler-tts-mini-jenny-30H
        device="cuda:0",
        torch_dtype=torch.bfloat16,
        buffer_duration_s=1.0,
        play_steps_in_s=0.5,
        print_time_to_first_token=False,
    ):
        """
        Initializes the Parler TTS engine.

        Args:
            model_name (str): Name of the Parler TTS model to use.
            device (str): Torch device to use (e.g., "cuda:0", "cpu").
            torch_dtype (torch.dtype): Torch data type to use.
            voice_prompt (str): Voice prompt for the model.
            play_steps_in_s (float): Duration in seconds for each play step.
        """
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.voice_prompt = voice_prompt
        self.play_steps_in_s = play_steps_in_s
        self.voice_parameters = {}
        self.buffer_duration_s = buffer_duration_s
        self.print_time_to_first_token = print_time_to_first_token

        self.initialize_model()

    def post_init(self):
        self.engine_name = "parler_tts"

    def initialize_model(self):
        """
        Loads the model and tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(self.device, dtype=self.torch_dtype)

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration information suitable for Parler TTS Engine.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
        """
        return pyaudio.paFloat32, 1, 44100  # Using 32-bit float, mono, 44100 Hz

    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.

        Returns:
            bool: True if synthesis starts successfully, False otherwise.
        """
        try:
            self._generate_and_queue_audio(text)
            return True
        except Exception as e:
            print(f"Error in synthesize: {e}")
            return False

    def _generate_and_queue_audio(self, text: str):
        """
        Generates audio from text, buffers audio for a set duration, and then streams it into the audio queue.

        Args:
            text (str): Text to synthesize.
        """
        start_time = time.time()
        frame_rate = self.model.audio_encoder.config.frame_rate
        sampling_rate = self.model.audio_encoder.config.sampling_rate

        play_steps = int(frame_rate * self.play_steps_in_s)
        streamer = ParlerTTSStreamer(self.model, device=self.device, play_steps=play_steps)

        if not self.voice_prompt:
            self.voice_prompt = "Female voice, calm and uplifting."

        inputs = self.tokenizer(self.voice_prompt, return_tensors="pt").to(self.device)
        prompt = self.tokenizer(text, return_tensors="pt").to(self.device)

        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "prompt_input_ids": prompt.input_ids,
            "attention_mask": inputs.attention_mask,
            "prompt_attention_mask": prompt.attention_mask,
            "streamer": streamer,
            "do_sample": True,
            "temperature": 1.0,
            "min_new_tokens": 10,
            **self.voice_parameters,  # Merge with any additional voice parameters
        }

        # Initialize variables for buffering
        audio_buffer = []
        buffer_length_s = 0.0
        generation_completed = False

        # Start the audio generation (blocking call)
        def generate_audio():
            self.model.generate(**generation_kwargs)

        # Start the generation in a separate thread
        generation_thread = Thread(target=generate_audio)
        generation_thread.start()

        # Process the streamer in the main thread
        while not generation_completed:
            try:
                new_audio = next(streamer)
                if new_audio.shape[0] == 0:
                    # Streamer signaled completion
                    generation_completed = True
                    break

                audio_chunk = new_audio
                audio_buffer.append(audio_chunk)
                buffer_length_s += new_audio.shape[0] / sampling_rate

                if buffer_length_s >= self.buffer_duration_s:
                    # Buffering complete, start streaming
                    break
            except StopIteration:
                # No more audio data
                generation_completed = True
                break

        # Queue the buffered audio chunks
        for buffered_chunk in audio_buffer:
            self.queue.put(buffered_chunk.tobytes())

        # Continue streaming the rest of the audio
        first_token = False
        while not generation_completed:
            try:
                new_audio = next(streamer)
                if new_audio.shape[0] == 0:
                    # Streamer signaled completion
                    generation_completed = True
                    break
                audio_chunk = new_audio
                if not first_token and self.print_time_to_first_token:
                    end_time = time.time()
                    print(f"Time to first token: {end_time - start_time:.2f} s")
                self.queue.put(audio_chunk.tobytes())
                first_token = True
            except StopIteration:
                generation_completed = True
                break

        # Ensure the generation thread has completed
        generation_thread.join()

    def get_voices(self):
        """
        Retrieves the voices available from Parler TTS.

        Returns:
            list: A list containing ParlerVoice objects representing each available voice.
        """
        return [
            ParlerVoice(
                "Laura",
                "Laura's voice is fast-paced with a slight rasp, recorded up close with no background noise. Her words come out quickly, creating a sense of urgency and intensity.",
            ),
            ParlerVoice(
                "John",
                "John's voice is deep and smooth, with a calm and reassuring tone. He speaks slowly and clearly, with a slight southern accent.",
            ),
        ]

    def set_voice(self, voice: Union[str, ParlerVoice]):
        """
        Sets the voice to be used for speech synthesis.

        Args:
            voice (Union[str, ParlerVoice]): The voice to be used for speech synthesis.
        """
        if isinstance(voice, ParlerVoice):
            self.voice_prompt = voice.description
        elif isinstance(voice, str):
            self.voice_prompt = voice
        else:
            raise ValueError("Voice must be a string or ParlerVoice instance.")

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets additional voice parameters for speech synthesis.

        Args:
            **voice_parameters: Arbitrary keyword arguments for voice parameters.
        """
        self.voice_parameters.update(voice_parameters)

    def shutdown(self):
        """
        Shuts down the engine.
        """
        pass  
