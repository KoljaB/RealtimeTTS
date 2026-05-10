from .base_engine import BaseEngine
from typing import Union
import os
import io
import wave
import pyaudio


class TypecastVoice:
    def __init__(self, name: str, voice_id: str, gender: str = "", age: str = "",
                 use_cases: list = None, models: list = None):
        self.name = name
        self.voice_id = voice_id
        self.gender = gender
        self.age = age
        self.use_cases = use_cases or []
        self.models = models or []

    def __repr__(self):
        return f"{self.name} ({self.voice_id})"


class TypecastEngine(BaseEngine):
    def __init__(
            self,
            voice_id: str = None,
            model: str = "ssfm-v30",
            tempo: float = None,
            pitch: int = None,
            volume: int = None,
            language: str = None,
            emotion_type: str = "preset",
            emotion_preset: str = "normal",
            emotion_intensity: float = None,
            seed: int = None,
            api_key: str = None,
            host: str = None,
            debug: bool = False,
            chunk_size: int = 4096,
        ):
        """
        Initializes a Typecast TTS engine using the official typecast-python SDK.

        Args:
            voice_id (str): Typecast voice ID. If unspecified, uses TYPECAST_VOICE_ID env var.
            model (str): Model version. "ssfm-v30" or "ssfm-v21". Defaults to "ssfm-v30".
            tempo (float): Speech speed multiplier (0.5–2.0).
            pitch (int): Pitch in semitones (-12 to +12).
            volume (int): Volume (0–200).
            language (str): Language code (ISO 639-3). Auto-detected if None.
            emotion_type (str): Emotion mode. "preset" or "smart" (ssfm-v30 only).
                "smart" infers emotion from context automatically.
            emotion_preset (str): Emotion preset when emotion_type is "preset".
                ssfm-v30: normal, happy, sad, angry, whisper, toneup, tonedown.
                ssfm-v21: normal, happy, sad, angry.
            emotion_intensity (float): Emotion intensity (0.0–2.0). Defaults to 1.0.
            seed (int): Random seed for reproducible output.
            api_key (str): Typecast API key. If unspecified, uses TYPECAST_API_KEY env var.
            host (str): API host URL override. If unspecified, the SDK uses
                TYPECAST_API_HOST env var or the default endpoint.
            debug (bool): Print debug information.
            chunk_size (int): Audio chunk size in bytes for streaming.
        """
        self.api_key = api_key or os.environ.get("TYPECAST_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Typecast API key is required. Provide it via api_key parameter "
                "or TYPECAST_API_KEY environment variable."
            )

        self.voice_id = voice_id or os.environ.get("TYPECAST_VOICE_ID")
        self.model = model
        self.tempo = tempo
        self.pitch = pitch
        self.volume = volume
        self.language = language
        self.emotion_type = emotion_type
        self.emotion_preset = emotion_preset
        self.emotion_intensity = emotion_intensity
        self.seed = seed
        self.host = host  # If None, SDK falls back to TYPECAST_API_HOST env var or default
        self.debug = debug
        self.chunk_size = chunk_size
        self._client = None

    def post_init(self):
        self.engine_name = "typecast"

    def _get_client(self):
        if self._client is None:
            try:
                from typecast import Typecast
            except ImportError as e:
                raise ImportError(
                    "typecast-python package is required. Install with:\n"
                    "pip install typecast-python"
                ) from e
            kwargs = {"api_key": self.api_key}
            if self.host:
                kwargs["host"] = self.host
            self._client = Typecast(**kwargs)
        return self._client

    def _build_prompt(self):
        """Builds the appropriate prompt object based on emotion_type and model."""
        from typecast.models import Prompt, PresetPrompt, SmartPrompt

        if self.model == "ssfm-v21":
            kwargs = {}
            if self.emotion_preset is not None:
                kwargs["emotion_preset"] = self.emotion_preset
            if self.emotion_intensity is not None:
                kwargs["emotion_intensity"] = self.emotion_intensity
            return Prompt(**kwargs) if kwargs else None

        # ssfm-v30
        if self.emotion_type == "smart":
            return SmartPrompt()
        else:
            kwargs = {}
            if self.emotion_preset is not None:
                kwargs["emotion_preset"] = self.emotion_preset
            if self.emotion_intensity is not None:
                kwargs["emotion_intensity"] = self.emotion_intensity
            return PresetPrompt(**kwargs)

    def get_stream_info(self):
        """
        Returns PyAudio stream configuration for Typecast (WAV: mono, 44100Hz, 16-bit).
        """
        return pyaudio.paInt16, 1, 44100

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        """
        Synthesizes text to audio using the typecast-python SDK and streams to queue.

        Args:
            text (str): Text to synthesize.
            sentence_count (int): The count of sentences synthesized so far.

        Returns:
            bool: True on success, False on failure.
        """
        super().synthesize(text, sentence_count)

        from typecast.models import TTSRequest, TTSModel, Output

        if not self.voice_id:
            print("[TypecastEngine] Error: voice_id is not set. "
                  "Pass voice_id to the constructor or set TYPECAST_VOICE_ID.")
            return False

        if self.debug:
            print(f"[TypecastEngine] Synthesizing: \"{text}\"")

        try:
            output_kwargs = {"audio_format": "wav"}
            if self.tempo is not None:
                output_kwargs["audio_tempo"] = self.tempo
            if self.pitch is not None:
                output_kwargs["audio_pitch"] = self.pitch
            if self.volume is not None:
                output_kwargs["volume"] = self.volume

            request = TTSRequest(
                text=text,
                voice_id=self.voice_id,
                model=TTSModel(self.model),
                language=self.language,
                prompt=self._build_prompt(),
                seed=self.seed,
                output=Output(**output_kwargs),
            )

            client = self._get_client()
            response = client.text_to_speech(request)

            with wave.open(io.BytesIO(response.audio_data), "rb") as wf:
                frames_per_chunk = self.chunk_size // (wf.getsampwidth() * wf.getnchannels())
                while not self.stop_synthesis_event.is_set():
                    data = wf.readframes(frames_per_chunk)
                    if not data:
                        break
                    self.queue.put(data)

        except Exception as e:
            print(f"[TypecastEngine] Error during synthesis: {e}")
            return False

        return True

    def get_voices(self) -> list:
        """
        Returns available Typecast voices via the SDK.
        """
        try:
            voices = self._get_client().voices_v2()
            return [
                TypecastVoice(
                    name=v.voice_name,
                    voice_id=v.voice_id,
                    gender=v.gender.value if v.gender else "",
                    age=v.age.value if v.age else "",
                    use_cases=v.use_cases or [],
                    models=[m.version for m in v.models] if v.models else [],
                )
                for v in voices
            ]
        except Exception as e:
            print(f"[TypecastEngine] Error fetching voices: {e}")
            return []

    def set_voice(self, voice: Union[str, TypecastVoice]):
        """
        Sets the voice for synthesis.

        Args:
            voice: Voice name/ID string or TypecastVoice object.
        """
        if isinstance(voice, TypecastVoice):
            self.voice_id = voice.voice_id
            return
        for v in self.get_voices():
            if voice.lower() in v.name.lower() or voice == v.voice_id:
                self.voice_id = v.voice_id
                return
        # Fallback: treat as raw voice_id
        self.voice_id = voice

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets voice parameters.

        Supported keys: tempo, pitch, volume, language, seed, model,
                        emotion_type, emotion_preset, emotion_intensity
        """
        for key, value in voice_parameters.items():
            if hasattr(self, key):
                setattr(self, key, value)
