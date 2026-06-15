"""
Pocket TTS Engine for RealtimeTTS

Requires:
- pip install pocket-tts torch

Pocket TTS is a lightweight 100M parameter TTS model from Kyutai Labs
that runs efficiently on CPU with ~6x real-time performance.

Features:
- CPU-optimized inference
- Voice cloning via WAV files
- ~200ms latency to first audio chunk
- English language only
"""

from .base_engine import BaseEngine
from queue import Queue
from pathlib import Path
from typing import Union, Optional
import numpy as np
import threading
import traceback
import pyaudio
import time
import os
import hashlib


def _patch_pocket_tts_serial_streaming(TTSModel) -> None:
    """Avoid PocketTTS' per-call decoder thread on Windows/Torch CPU.

    PocketTTS' public streaming path creates a fresh decoder worker thread for
    every generation. On Windows CPU builds we observed native Torch memory
    growth from repeated short generations even though Python tensors and
    tracemalloc stayed flat. Running the same generation/decode sequence
    serially in one long-lived engine worker avoids that retention.
    """

    if getattr(TTSModel, "_realtimetts_serial_streaming_patch", False):
        return

    try:
        import copy
        import torch
        from pocket_tts.modules.stateful_module import increment_steps, init_states
    except ImportError:
        return

    @torch.no_grad
    def _generate_audio_stream_short_text(
        self,
        model_state: dict,
        text_to_generate: str,
        frames_after_eos: int,
        copy_state: bool,
    ):
        if copy_state:
            model_state = copy.deepcopy(model_state)

        prepared = self.flow_lm.conditioner.prepare(text_to_generate)
        token_count = prepared.tokens.shape[1]
        max_gen_len = self._estimate_max_gen_len(token_count)
        mimi_steps_per_latent = int(self.mimi.encoder_frame_rate / self.mimi.frame_rate)
        mimi_sequence_length = max_gen_len * mimi_steps_per_latent

        current_end = self._flow_lm_current_end(model_state)
        required_len = current_end + token_count + max_gen_len
        self._expand_kv_cache(model_state, sequence_length=required_len)
        self._run_flow_lm_and_increment_step(
            model_state=model_state,
            text_tokens=prepared.tokens,
        )

        mimi_state = init_states(
            self.mimi,
            batch_size=1,
            sequence_length=mimi_sequence_length,
        )
        backbone_input = torch.full(
            (1, 1, self.flow_lm.ldim),
            fill_value=float("NaN"),
            device=next(iter(self.flow_lm.parameters())).device,
            dtype=self.flow_lm.dtype,
        )
        eos_step = None
        for generation_step in range(max_gen_len):
            next_latent, is_eos = self._run_flow_lm_and_increment_step(
                model_state=model_state,
                backbone_input_latents=backbone_input,
            )
            if is_eos.item() and eos_step is None:
                eos_step = generation_step
            if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                break

            mimi_decoding_input = next_latent * self.flow_lm.emb_std + self.flow_lm.emb_mean
            transposed = mimi_decoding_input.transpose(-1, -2)
            quantized = self.mimi.quantizer(transposed)
            audio_frame = self.mimi.decode_from_latent(quantized, mimi_state)
            increment_steps(self.mimi, mimi_state, increment=mimi_steps_per_latent)
            yield audio_frame[0, 0]
            backbone_input = next_latent

    TTSModel._generate_audio_stream_short_text = _generate_audio_stream_short_text
    TTSModel._realtimetts_serial_streaming_patch = True


class PocketTTSVoice:
    """Represents a voice for the PocketTTS engine."""

    # Built-in voices available in pocket-tts
    BUILTIN_VOICES = [
        "alba", "marius", "javert", "jean",
        "fantine", "cosette", "eponine", "azelma"
    ]

    def __init__(
        self,
        name: str,
        audio_prompt_path: Optional[str] = None,
        state_path: Optional[str] = None
    ):
        """
        Initialize a PocketTTS voice.

        Args:
            name: Voice identifier (built-in name or custom name for cloned voice)
            audio_prompt_path: Path to WAV file for voice cloning (optional)
        """
        self.name = name
        self.audio_prompt_path = audio_prompt_path
        self.state_path = state_path
        self.is_builtin = name in self.BUILTIN_VOICES and audio_prompt_path is None

    def __repr__(self):
        if self.state_path:
            return f"PocketTTSVoice(name='{self.name}', state='{self.state_path}')"
        if self.audio_prompt_path:
            return f"PocketTTSVoice(name='{self.name}', cloned_from='{self.audio_prompt_path}')"
        return f"PocketTTSVoice(name='{self.name}', builtin={self.is_builtin})"


class PocketTTSEngine(BaseEngine):
    """
    A text-to-speech engine using Kyutai Labs' Pocket TTS model.

    Pocket TTS is a lightweight 100M parameter model optimized for CPU inference,
    achieving ~6x real-time performance with ~200ms latency to first audio chunk.

    Example usage:
        engine = PocketTTSEngine(voice="alba")
        engine.synthesize("Hello, world!")

    For voice cloning:
        voice = PocketTTSVoice(name="custom", audio_prompt_path="path/to/voice.wav")
        engine = PocketTTSEngine(voice=voice)
    """

    def __init__(
        self,
        voice: Union[str, PocketTTSVoice] = "alba",
        trim_silence: bool = True,
        silence_threshold: float = 0.005,
        extra_start_ms: int = 15,
        extra_end_ms: int = 15,
        fade_in_ms: int = 10,
        fade_out_ms: int = 10,
        streaming: bool = True,
        max_tokens: int = 50,
        frames_after_eos: Optional[int] = None,
        model_config: Optional[str] = None,
        device: str = "cpu",
        voice_cache_dir: Optional[str] = None,
        cache_voice_states: bool = True,
        debug: bool = False
    ):
        """
        Initialize the PocketTTS engine.

        Args:
            voice: Voice to use (built-in name string or PocketTTSVoice object)
            trim_silence: Whether to trim silence from audio output
            silence_threshold: Threshold for silence detection
            extra_start_ms: Extra milliseconds to trim from start
            extra_end_ms: Extra milliseconds to trim from end
            fade_in_ms: Fade-in duration in milliseconds
            fade_out_ms: Fade-out duration in milliseconds
            streaming: Whether to use Pocket TTS' streaming generator
            max_tokens: Maximum generation tokens passed to Pocket TTS
            frames_after_eos: Optional trailing frames after EOS for Pocket TTS
            device: Torch device for the Pocket TTS model, for example "cpu" or "cuda"
            debug: Enable debug output
        """
        super().__init__()
        self.engine_name = "pocket_tts"
        self.debug = debug
        self.queue = Queue()

        # Silence trimming settings
        self.trim_silence = trim_silence
        self.silence_threshold = silence_threshold
        self.extra_start_ms = extra_start_ms
        self.extra_end_ms = extra_end_ms
        self.fade_in_ms = fade_in_ms
        self.fade_out_ms = fade_out_ms
        self.streaming = streaming
        self.max_tokens = max_tokens
        self.frames_after_eos = frames_after_eos
        self.model_config = model_config
        self.device = device
        self.voice_cache_dir = Path(voice_cache_dir) if voice_cache_dir else None
        self.cache_voice_states = cache_voice_states

        # Initialize the model
        self.model = None
        self.sample_rate = None
        self._voice_states = {}  # Cache for voice states
        self.current_voice = None
        self.current_voice_state = None

        self._load_model()
        self.set_voice(voice)

        if self.debug:
            print(f"[PocketTTSEngine] Initialized with voice: {self.current_voice}")


    def post_init(self):
        """Restore Pocket state after BaseEngine's metaclass post-init hook."""
        self.engine_name = "pocket_tts"
        self._synthesis_requests = Queue()
        self._synthesis_worker_thread = threading.Thread(
            target=self._synthesis_worker_loop,
            name="PocketTTSSynthesisWorker",
            daemon=True,
        )
        self._synthesis_worker_thread.start()

    def _synthesis_worker_loop(self):
        while True:
            job = self._synthesis_requests.get()
            if job is None:
                self._synthesis_requests.task_done()
                break

            text, sentence_count, response_queue = job
            try:
                response_queue.put((self._synthesize_impl(text, sentence_count), None))
            except Exception as exc:
                response_queue.put((False, exc))
            finally:
                self._synthesis_requests.task_done()

    def _synthesize_on_worker(self, text: str, sentence_count: int = 0) -> bool:
        worker = getattr(self, "_synthesis_worker_thread", None)
        if worker is None or threading.current_thread() is worker:
            return self._synthesize_impl(text, sentence_count)

        response_queue = Queue(maxsize=1)
        self._synthesis_requests.put((text, sentence_count, response_queue))
        success, error = response_queue.get()
        if error is not None:
            raise error
        return success

    def _load_model(self):
        """Load the Pocket TTS model."""
        try:
            from pocket_tts import TTSModel
            _patch_pocket_tts_serial_streaming(TTSModel)

            if self.debug:
                print("[PocketTTSEngine] Loading Pocket TTS model...")

            model_config = self.model_config or self._auto_voice_cloning_config()
            if model_config:
                self.model = TTSModel.load_model(config=str(model_config))
            else:
                self.model = TTSModel.load_model()
            if self.device:
                self.model.to(self.device)
            self.sample_rate = self.model.sample_rate

            if self.debug:
                print(
                    f"[PocketTTSEngine] Model loaded on {self.device}. "
                    f"Sample rate: {self.sample_rate}"
                )

        except ImportError:
            raise ImportError(
                "pocket-tts is not installed. Install it with: pip install pocket-tts"
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to load Pocket TTS model: {e}")

    def _auto_voice_cloning_config(self) -> Optional[Path]:
        if self.voice_cache_dir is None:
            return None
        hf_home = Path(os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface")
        snapshots_dir = hf_home / "hub" / "models--kyutai--pocket-tts" / "snapshots"
        if not snapshots_dir.exists():
            return None
        candidates = sorted(
            snapshots_dir.glob("*/languages/english/model.safetensors"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        if not candidates:
            return None
        model_path = candidates[0]
        config_path = self.voice_cache_dir / "english_local_clone.yaml"
        try:
            from pocket_tts.utils.config import CONFIGS_DIR

            source_config = CONFIGS_DIR / "english.yaml"
            text = source_config.read_text(encoding="utf-8")
            lines = []
            for line in text.splitlines():
                if line.startswith("weights_path: "):
                    lines.append(f"weights_path: {str(model_path).replace(os.sep, '/')}")
                else:
                    lines.append(line)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            if self.debug:
                print(f"[PocketTTSEngine] Using local voice-cloning config: {config_path}")
            return config_path
        except Exception:
            if self.debug:
                print("[PocketTTSEngine] Could not create local voice-cloning config")
            return None

    def _get_voice_state(self, voice: Union[str, PocketTTSVoice]):
        """
        Get or create a voice state for the given voice.

        Args:
            voice: Voice identifier or PocketTTSVoice object

        Returns:
            Voice state object for use with generate_audio
        """
        if isinstance(voice, PocketTTSVoice):
            voice_name = voice.name
            audio_path = voice.audio_prompt_path
            state_path = voice.state_path
        else:
            voice_name = voice
            audio_path = None
            state_path = None

        # Check cache
        cache_key = f"{voice_name}:{state_path or audio_path or 'builtin'}"
        if cache_key in self._voice_states:
            if self.debug:
                print(f"[PocketTTSEngine] Using cached voice state for: {cache_key}")
            return self._voice_states[cache_key]

        # Create new voice state
        if state_path:
            if not os.path.exists(state_path):
                raise FileNotFoundError(f"Voice state file not found: {state_path}")
            if self.debug:
                print(f"[PocketTTSEngine] Loading voice state from: {state_path}")
            voice_state = self.model.get_state_for_audio_prompt(state_path)
        elif audio_path:
            # Voice cloning from WAV file
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Voice audio file not found: {audio_path}")

            if self.debug:
                print(f"[PocketTTSEngine] Creating voice state from audio: {audio_path}")

            cached_state_path = self._cached_state_path(voice_name, audio_path)
            if cached_state_path is not None and cached_state_path.exists():
                if self.debug:
                    print(f"[PocketTTSEngine] Loading cached voice state: {cached_state_path}")
                voice_state = self.model.get_state_for_audio_prompt(str(cached_state_path))
            else:
                prompt_path = self._pcm16_prompt_path(voice_name, audio_path)
                voice_state = self.model.get_state_for_audio_prompt(str(prompt_path))
                if cached_state_path is not None:
                    self._export_voice_state(voice_state, cached_state_path)
        else:
            # Built-in voice
            if voice_name not in PocketTTSVoice.BUILTIN_VOICES:
                raise ValueError(
                    f"Unknown voice: {voice_name}. "
                    f"Available voices: {PocketTTSVoice.BUILTIN_VOICES}"
                )

            if self.debug:
                print(f"[PocketTTSEngine] Creating voice state for built-in voice: {voice_name}")

            voice_state = self.model.get_state_for_audio_prompt(voice_name)

        # Cache and return
        self._voice_states[cache_key] = voice_state
        return voice_state

    def _cached_state_path(self, voice_name: str, audio_path: str) -> Optional[Path]:
        if not self.cache_voice_states or self.voice_cache_dir is None:
            return None
        source = Path(audio_path)
        try:
            stat = source.stat()
            cache_id = f"{source.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            cache_id = str(source)
        digest = hashlib.sha256(cache_id.encode("utf-8")).hexdigest()[:16]
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in voice_name)
        return self.voice_cache_dir / f"{safe_name}_{digest}.safetensors"

    def _pcm16_prompt_path(self, voice_name: str, audio_path: str) -> str:
        if self.voice_cache_dir is None:
            return audio_path
        source = Path(audio_path)
        try:
            stat = source.stat()
            cache_id = f"{source.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            cache_id = str(source)
        digest = hashlib.sha256(cache_id.encode("utf-8")).hexdigest()[:16]
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in voice_name)
        pcm_path = self.voice_cache_dir / "prompt_wavs" / f"{safe_name}_{digest}.wav"
        if pcm_path.exists():
            return str(pcm_path)
        pcm_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from scipy.io import wavfile

            sample_rate, data = wavfile.read(str(source))
            audio = np.asarray(data)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if np.issubdtype(audio.dtype, np.floating):
                audio_float = np.clip(audio.astype(np.float32), -1.0, 1.0)
            else:
                info = np.iinfo(audio.dtype)
                scale = float(max(abs(info.min), abs(info.max)))
                audio_float = audio.astype(np.float32) / scale
            audio_int16 = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
            wavfile.write(str(pcm_path), sample_rate, audio_int16)
            return str(pcm_path)
        except Exception:
            if self.debug:
                print(f"[PocketTTSEngine] Could not convert prompt WAV to PCM16: {audio_path}")
            return audio_path

    def _export_voice_state(self, voice_state, state_path: Path) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from pocket_tts import export_model_state

            export_model_state(voice_state, state_path)
            if self.debug:
                print(f"[PocketTTSEngine] Cached voice state: {state_path}")
        except Exception:
            if self.debug:
                print(f"[PocketTTSEngine] Could not cache voice state: {state_path}")

    def get_stream_info(self):
        """
        Returns the audio stream configuration for PyAudio.

        Returns:
            tuple: (format, channels, sample_rate)
        """
        # Pocket TTS typically uses 24kHz sample rate
        sample_rate = self.sample_rate if self.sample_rate else 24000
        return (pyaudio.paInt16, 1, sample_rate)

    def _to_numpy_audio(self, audio) -> np.ndarray:
        """Convert a Pocket TTS tensor or array to flat float32 numpy audio."""
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu()
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        audio_float32 = np.asarray(audio, dtype=np.float32)
        if audio_float32.ndim > 1:
            audio_float32 = audio_float32.squeeze()
        return audio_float32.reshape(-1)

    def _queue_audio(self, audio_float32: np.ndarray) -> int:
        if audio_float32.size == 0:
            return 0
        audio_float32 = np.clip(audio_float32, -1.0, 1.0)
        audio_int16 = (audio_float32 * 32767).astype(np.int16).tobytes()
        self.audio_duration += len(audio_float32) / self.sample_rate
        self.queue.put(audio_int16)
        return len(audio_int16)

    def _synthesize_streaming(self, text: str, start_time: float) -> bool:
        chunk_count = 0
        sample_count = 0
        for audio_chunk in self.model.generate_audio_stream(
            self.current_voice_state,
            text,
            max_tokens=self.max_tokens,
            frames_after_eos=self.frames_after_eos,
        ):
            if self.stop_synthesis_event.is_set():
                if self.debug:
                    print("[PocketTTSEngine] Streaming synthesis stopped")
                return True
            audio_float32 = self._to_numpy_audio(audio_chunk)
            if self.trim_silence:
                audio_float32 = self._trim_silence(
                    audio_float32,
                    sample_rate=self.sample_rate,
                    silence_threshold=self.silence_threshold,
                    extra_start_ms=self.extra_start_ms,
                    extra_end_ms=self.extra_end_ms,
                    fade_in_ms=self.fade_in_ms,
                    fade_out_ms=self.fade_out_ms,
                )
            queued_bytes = self._queue_audio(audio_float32)
            if queued_bytes:
                chunk_count += 1
                sample_count += len(audio_float32)

        if self.debug:
            duration = time.time() - start_time
            audio_length_seconds = sample_count / self.sample_rate
            print(
                f"[PocketTTSEngine] Streaming synthesis completed in {duration:.3f}s "
                f"({audio_length_seconds:.2f}s of audio, {chunk_count} chunks)"
            )
        return True

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        return self._synthesize_on_worker(text, sentence_count)

    def _synthesize_impl(self, text: str, sentence_count: int = 0) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
            sentence_count (int): The count of sentences synthesized so far, used for tracking progress.

        Returns:
            bool: True if successful, False otherwise.
        """
        super().synthesize(text, sentence_count)

        if self.stop_synthesis_event.is_set():
            return True

        if not text or not text.strip():
            if self.debug:
                print("[PocketTTSEngine] Empty text, skipping synthesis")
            return True

        start_time = time.time()

        try:
            if self.model is None:
                print("[PocketTTSEngine] Model not loaded")
                return False

            if self.current_voice_state is None:
                print("[PocketTTSEngine] No voice set")
                return False

            if self.debug:
                print(f"[PocketTTSEngine] Synthesizing: '{text[:50]}...'")

            if self.streaming and hasattr(self.model, "generate_audio_stream"):
                return self._synthesize_streaming(text, start_time)

            # Generate audio
            audio_tensor = self.model.generate_audio(
                self.current_voice_state,
                text,
                max_tokens=self.max_tokens,
                frames_after_eos=self.frames_after_eos,
            )
            audio_float32 = self._to_numpy_audio(audio_tensor)

            # Trim silence if enabled
            if self.trim_silence:
                audio_float32 = self._trim_silence(
                    audio_float32,
                    sample_rate=self.sample_rate,
                    silence_threshold=self.silence_threshold,
                    extra_start_ms=self.extra_start_ms,
                    extra_end_ms=self.extra_end_ms,
                    fade_in_ms=self.fade_in_ms,
                    fade_out_ms=self.fade_out_ms,
                )

            self._queue_audio(audio_float32)

            if self.debug:
                duration = time.time() - start_time
                audio_length_seconds = len(audio_float32) / self.sample_rate
                print(f"[PocketTTSEngine] Synthesis completed in {duration:.3f}s "
                      f"({audio_length_seconds:.2f}s of audio)")

            return True

        except Exception as e:
            traceback.print_exc()
            print(f"[PocketTTSEngine] Error during synthesis: {e}")
            return False

    def get_voices(self) -> list:
        """
        Get list of available voices.

        Returns:
            list: List of PocketTTSVoice objects for built-in voices
        """
        return [PocketTTSVoice(name) for name in PocketTTSVoice.BUILTIN_VOICES]

    def set_voice(self, voice: Union[str, PocketTTSVoice]):
        """
        Set the current voice for synthesis.

        Args:
            voice: Voice identifier (string) or PocketTTSVoice object
        """
        try:
            if isinstance(voice, str):
                # Check if it's a built-in voice
                if voice in PocketTTSVoice.BUILTIN_VOICES:
                    self.current_voice = PocketTTSVoice(voice)
                else:
                    # Assume it's a path to a WAV file for cloning
                    if os.path.exists(voice):
                        if str(voice).lower().endswith(".safetensors"):
                            self.current_voice = PocketTTSVoice(
                                name=os.path.basename(voice),
                                state_path=voice
                            )
                        else:
                            self.current_voice = PocketTTSVoice(
                                name=os.path.basename(voice),
                                audio_prompt_path=voice
                            )
                    else:
                        raise ValueError(
                            f"Unknown voice: {voice}. "
                            f"Available voices: {PocketTTSVoice.BUILTIN_VOICES}"
                        )
            else:
                self.current_voice = voice

            # Get/create voice state
            self.current_voice_state = self._get_voice_state(self.current_voice)

            if self.debug:
                print(f"[PocketTTSEngine] Voice set to: {self.current_voice}")

        except Exception as e:
            traceback.print_exc()
            print(f"[PocketTTSEngine] Error setting voice: {e}")
            raise

    def set_voice_parameters(self, **voice_parameters):
        """
        Set voice parameters.

        Currently Pocket TTS doesn't support runtime voice parameters
        like speed adjustment.

        Args:
            **voice_parameters: Voice parameters (currently unused)
        """
        if self.debug:
            print(f"[PocketTTSEngine] set_voice_parameters called with: {voice_parameters}")
        for name in ("streaming", "max_tokens", "frames_after_eos"):
            if name in voice_parameters:
                setattr(self, name, voice_parameters[name])

    def shutdown(self):
        """Shutdown the engine and release resources."""
        if self.debug:
            print("[PocketTTSEngine] Shutdown called")

        worker = getattr(self, "_synthesis_worker_thread", None)
        requests = getattr(self, "_synthesis_requests", None)
        if requests is not None and worker is not None and worker.is_alive():
            requests.put(None)
            if threading.current_thread() is not worker:
                worker.join(timeout=5.0)

        self._synthesis_worker_thread = None

        # Clear caches
        self._voice_states.clear()
        self.current_voice_state = None
        self.model = None
