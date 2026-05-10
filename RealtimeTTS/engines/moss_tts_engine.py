"""MOSS-TTS engine for RealtimeTTS.

Primary target: MOSS-TTS-Nano from https://github.com/OpenMOSS/MOSS-TTS-Nano.
The Nano wrapper supports the official ONNX runtime and torch runtime. It keeps
reference audio explicit so Zaphod benchmarks can use one neutral cloning
reference instead of registering a whole emotion folder.
"""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .base_engine import BaseEngine


class MossTTSVoice:
    """Reference audio or built-in voice configuration for MOSS-TTS."""

    def __init__(
        self,
        name: str = "neutral",
        prompt_audio_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        voice: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        prompt_wav_path: Optional[str] = None,
    ):
        audio_path = prompt_audio_path or ref_audio_path or prompt_wav_path
        if audio_path and not os.path.exists(audio_path):
            raise FileNotFoundError("MOSS-TTS reference audio not found: %s" % audio_path)
        self.name = name
        self.prompt_audio_path = audio_path
        self.prompt_text = (prompt_text or ref_text or "").strip() or None
        self.voice = voice

    def __repr__(self):
        return "MossTTSVoice(name=%r, prompt_audio_path=%r, voice=%r)" % (
            self.name,
            self.prompt_audio_path,
            self.voice,
        )


class MossTTSEngine(BaseEngine):
    """RealtimeTTS wrapper for MOSS-TTS-Nano ONNX and torch runtimes."""

    def __init__(
        self,
        voice: Optional[MossTTSVoice] = None,
        variant: str = "nano",
        backend: str = "onnx",
        model_repo: str = "OpenMOSS-Team/MOSS-TTS-Nano",
        model_path: Optional[str] = None,
        audio_tokenizer_repo: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        audio_tokenizer_path: Optional[str] = None,
        onnx_model_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "auto",
        attn_implementation: str = "auto",
        execution_provider: str = "cpu",
        streaming: bool = True,
        sample_rate: int = 48000,
        channels: int = 2,
        voice_name: str = "Junhao",
        output_dir: Optional[str] = None,
        max_new_frames: int = 375,
        voice_clone_max_text_tokens: int = 75,
        voice_clone_max_memory_per_sample_gb: float = 1.0,
        tts_max_batch_size: int = 0,
        codec_max_batch_size: int = 0,
        sample_mode: str = "fixed",
        do_sample: bool = True,
        text_temperature: float = 1.0,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 0.8,
        audio_top_p: float = 0.95,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.2,
        stream_decode_frames: int = 1,
        enable_wetext_processing: bool = False,
        enable_normalize_tts_text: bool = True,
        seed: Optional[int] = None,
        realtime_temperature: float = 0.8,
        realtime_top_p: float = 0.6,
        realtime_top_k: int = 30,
        realtime_repetition_penalty: float = 1.1,
        realtime_repetition_window: int = 50,
        realtime_chunk_duration: int = 8,
    ):
        del (
            realtime_temperature,
            realtime_top_p,
            realtime_top_k,
            realtime_repetition_penalty,
            realtime_repetition_window,
            realtime_chunk_duration,
        )
        self.variant = self._normalize_variant(variant)
        self.backend = self._normalize_backend(backend)
        self.model_repo = model_repo
        self.model_path = model_path
        self.audio_tokenizer_repo = audio_tokenizer_repo
        self.audio_tokenizer_path = audio_tokenizer_path
        self.onnx_model_dir = onnx_model_dir
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.execution_provider = execution_provider
        self.streaming = streaming
        self.sampling_rate = int(sample_rate)
        self.channels = int(channels)
        self.voice_name = voice_name
        self.output_dir = Path(output_dir or "test_outputs/moss_tts_internal").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_new_frames = int(max_new_frames)
        self.voice_clone_max_text_tokens = int(voice_clone_max_text_tokens)
        self.voice_clone_max_memory_per_sample_gb = float(voice_clone_max_memory_per_sample_gb)
        self.tts_max_batch_size = int(tts_max_batch_size)
        self.codec_max_batch_size = int(codec_max_batch_size)
        self.sample_mode = sample_mode
        self.do_sample = bool(do_sample)
        self.text_temperature = float(text_temperature)
        self.text_top_p = float(text_top_p)
        self.text_top_k = int(text_top_k)
        self.audio_temperature = float(audio_temperature)
        self.audio_top_p = float(audio_top_p)
        self.audio_top_k = int(audio_top_k)
        self.audio_repetition_penalty = float(audio_repetition_penalty)
        self.stream_decode_frames = max(1, int(stream_decode_frames))
        self.enable_wetext_processing = bool(enable_wetext_processing)
        self.enable_normalize_tts_text = bool(enable_normalize_tts_text)
        self.seed = seed
        self._runtime = None
        self._voices: dict[str, MossTTSVoice] = {}
        self._current_voice = None
        self._dll_directory_handles = []

        self._init_runtime()
        if voice is not None:
            self.set_voice(voice)

    def post_init(self):
        self.engine_name = "moss-tts-nano"

    @staticmethod
    def _normalize_variant(variant: str) -> str:
        normalized = (variant or "nano").strip().lower().replace("_", "-")
        if normalized in {"nano", "moss-tts-nano", "moss-nano"}:
            return "nano"
        if normalized in {"realtime", "moss-tts-realtime", "moss-realtime"}:
            return "realtime"
        raise ValueError("MOSS variant must be 'nano' or 'realtime'.")

    @staticmethod
    def _normalize_backend(backend: str) -> str:
        normalized = (backend or "onnx").strip().lower().replace("_", "-")
        if normalized in {"onnx", "torch"}:
            return normalized
        if normalized in {"realtime", "moss-realtime"}:
            return "realtime"
        raise ValueError("MOSS backend must be one of: onnx, torch, realtime.")

    def _install_hint(self) -> str:
        return (
            "Install MOSS-TTS-Nano in the engine venv, then install editable RealtimeTTS:\n"
            "  pip install -e third_party/MOSS-TTS-Nano\n"
            "  pip install -e D:\\Projekte\\TTS\\RealtimeTTS\\RealtimeTTS_Work\\RealtimeTTS"
        )

    def _init_runtime(self) -> None:
        if self.variant != "nano" or self.backend == "realtime":
            raise ImportError(
                "MOSS-TTS-Realtime is not implemented in this RealtimeTTS wrapper yet. "
                "Use variant='nano' with backend='onnx' or backend='torch' first."
            )
        if self.backend == "onnx":
            self._init_onnx_runtime()
            return
        self._init_torch_runtime()

    def _init_onnx_runtime(self) -> None:
        if self._execution_provider_is_cuda():
            self._prepare_onnx_cuda_dlls()

        try:
            from onnx_tts_runtime import OnnxTtsRuntime
        except ImportError as exc:
            raise ImportError(
                "Failed to import MOSS-TTS-Nano ONNX runtime. %s" % self._install_hint()
            ) from exc

        self._runtime = OnnxTtsRuntime(
            model_dir=self.onnx_model_dir,
            thread_count=4,
            max_new_frames=self.max_new_frames,
            do_sample=self.do_sample,
            sample_mode=self.sample_mode,
            execution_provider=self.execution_provider,
            output_dir=self.output_dir,
        )
        defaults = self._runtime.manifest["generation_defaults"]
        defaults["text_temperature"] = self.text_temperature
        defaults["text_top_p"] = self.text_top_p
        defaults["text_top_k"] = self.text_top_k
        defaults["audio_temperature"] = self.audio_temperature
        defaults["audio_top_p"] = self.audio_top_p
        defaults["audio_top_k"] = self.audio_top_k
        defaults["audio_repetition_penalty"] = self.audio_repetition_penalty
        self.sampling_rate = int(self._runtime.codec_meta["codec_config"]["sample_rate"])
        self.channels = int(self._runtime.codec_meta["codec_config"]["channels"])
        logging.info(
            "Loaded MOSS-TTS-Nano ONNX runtime provider=%s sample_rate=%s channels=%s",
            self.execution_provider,
            self.sampling_rate,
            self.channels,
        )

    def _execution_provider_is_cuda(self) -> bool:
        normalized = str(self.execution_provider or "").strip().lower()
        return normalized in {"cuda", "gpu", "cudaexecutionprovider"}

    def _prepare_onnx_cuda_dlls(self) -> None:
        if os.name != "nt":
            return

        candidate_dirs = self._cuda_dll_candidate_dirs()
        if candidate_dirs:
            self._register_dll_directories(candidate_dirs)

        try:
            import onnxruntime as ort
        except ImportError:
            return

        preload_dlls = getattr(ort, "preload_dlls", None)
        if not callable(preload_dlls):
            return

        cuda_dirs = [
            path
            for path in candidate_dirs
            if self._contains_any(path, ("cudart64_12.dll", "cublas64_12.dll"))
        ]
        cudnn_dirs = [
            path for path in candidate_dirs if self._contains_any(path, ("cudnn64_9.dll",))
        ]

        for index, directory in enumerate(cuda_dirs):
            try:
                preload_dlls(cuda=True, cudnn=False, msvc=index == 0, directory=str(directory))
                logging.info("Preloaded ONNX Runtime CUDA DLLs from %s", directory)
            except Exception as exc:  # pragma: no cover - depends on host CUDA layout.
                logging.debug("Failed to preload ONNX Runtime CUDA DLLs from %s: %s", directory, exc)

        for directory in cudnn_dirs:
            try:
                preload_dlls(cuda=False, cudnn=True, msvc=False, directory=str(directory))
                logging.info("Preloaded ONNX Runtime cuDNN DLLs from %s", directory)
            except Exception as exc:  # pragma: no cover - depends on host CUDA layout.
                logging.debug("Failed to preload ONNX Runtime cuDNN DLLs from %s: %s", directory, exc)

    def _cuda_dll_candidate_dirs(self) -> list[Path]:
        candidates = []

        for env_name in ("MOSS_CUDA_DLL_DIRS", "ORT_CUDA_DLL_DIRS"):
            for item in os.environ.get(env_name, "").split(os.pathsep):
                if item.strip():
                    candidates.append(Path(item.strip()))

        for env_name in ("CUDA_PATH", "CUDA_HOME"):
            env_path = os.environ.get(env_name)
            if env_path:
                candidates.append(Path(env_path) / "bin")

        cudnn_path = os.environ.get("CUDNN_PATH")
        if cudnn_path:
            path = Path(cudnn_path)
            candidates.extend([path, path / "bin"])

        for root in (
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
            Path("C:/Program Files/NVIDIA/CUDNN"),
        ):
            if root.exists():
                candidates.extend(root.glob("v*/bin"))
                candidates.extend(root.glob("v*/bin/*"))

        deduped = []
        seen = set()
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                continue
            if not resolved.is_dir():
                continue
            key = str(resolved).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(resolved)
        return deduped

    def _register_dll_directories(self, directories: list[Path]) -> None:
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if not callable(add_dll_directory):
            return
        for directory in directories:
            try:
                self._dll_directory_handles.append(add_dll_directory(str(directory)))
                logging.debug("Registered DLL directory for MOSS-TTS ONNX CUDA: %s", directory)
            except OSError as exc:  # pragma: no cover - depends on host permissions.
                logging.debug("Failed to register DLL directory %s: %s", directory, exc)

    @staticmethod
    def _contains_any(directory: Path, filenames: tuple[str, ...]) -> bool:
        lowered = {name.lower() for name in filenames}
        try:
            return any(path.name.lower() in lowered for path in directory.iterdir())
        except OSError:
            return False

    def _init_torch_runtime(self) -> None:
        try:
            from moss_tts_nano_runtime import NanoTTSService
        except ImportError as exc:
            raise ImportError(
                "Failed to import MOSS-TTS-Nano torch runtime. %s" % self._install_hint()
            ) from exc

        checkpoint = self.model_path or self.model_repo
        tokenizer = self.audio_tokenizer_path or self.audio_tokenizer_repo
        self._runtime = NanoTTSService(
            checkpoint_path=checkpoint,
            audio_tokenizer_path=tokenizer,
            device=self.device,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            output_dir=self.output_dir,
        )
        logging.info(
            "Loaded MOSS-TTS-Nano torch runtime checkpoint=%s tokenizer=%s device=%s",
            checkpoint,
            tokenizer,
            self.device,
        )

    def get_stream_info(self):
        import pyaudio

        return pyaudio.paInt16, self.channels, self.sampling_rate

    def get_voices(self):
        return list(self._voices.values())

    def set_voice(self, voice: Union[str, MossTTSVoice]):
        if isinstance(voice, MossTTSVoice):
            self._voices[voice.name] = voice
            self._current_voice = voice
            return
        if voice in self._voices:
            self._current_voice = self._voices[voice]
            return
        self._current_voice = MossTTSVoice(name=str(voice), voice=str(voice))

    def set_voice_parameters(self, **voice_parameters):
        valid_params = {
            "streaming",
            "max_new_frames",
            "voice_clone_max_text_tokens",
            "sample_mode",
            "do_sample",
            "text_temperature",
            "text_top_p",
            "text_top_k",
            "audio_temperature",
            "audio_top_p",
            "audio_top_k",
            "audio_repetition_penalty",
            "stream_decode_frames",
            "enable_wetext_processing",
            "enable_normalize_tts_text",
        }
        for param, value in voice_parameters.items():
            if param in valid_params:
                setattr(self, param, value)

    def _voice(self) -> MossTTSVoice:
        if self._current_voice is not None:
            return self._current_voice
        return MossTTSVoice(name=self.voice_name, voice=self.voice_name)

    def _output_path(self) -> Path:
        name = "moss_%s_%s.wav" % (time.strftime("%Y%m%d_%H%M%S"), uuid.uuid4().hex[:8])
        return self.output_dir / name

    def _audio_array_to_pcm16(self, waveform) -> bytes:
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim == 0:
            return b""
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        elif audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
            audio = audio.T
        elif audio.ndim > 2:
            audio = np.squeeze(audio)
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)
            elif audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
                audio = audio.T
        if audio.ndim != 2 or audio.size == 0:
            return b""

        current_channels = int(audio.shape[1])
        if current_channels != self.channels:
            if current_channels == 1 and self.channels > 1:
                audio = np.repeat(audio, self.channels, axis=1)
            elif current_channels > 1 and self.channels == 1:
                audio = audio.mean(axis=1, keepdims=True)
            else:
                audio = audio[:, : self.channels]
        clipped = np.clip(audio, -1.0, 1.0)
        return np.round(clipped * 32767.0).astype(np.int16).tobytes()

    def _queue_audio(self, waveform) -> None:
        audio_bytes = self._audio_array_to_pcm16(waveform)
        if audio_bytes:
            self.queue.put(audio_bytes)

    def _onnx_prompt_audio_codes(self, voice: MossTTSVoice):
        return self._runtime.resolve_prompt_audio_codes(
            voice=voice.voice or self.voice_name,
            prompt_audio_path=voice.prompt_audio_path,
        )

    def _decode_onnx_audio(self, audio, audio_length: int):
        if audio_length <= 0:
            return None
        channels = int(audio.shape[1])
        return np.stack(
            [audio[0, channel_index, :audio_length] for channel_index in range(channels)],
            axis=1,
        ).astype(np.float32, copy=False)

    def _synthesize_onnx_text_chunk(self, text: str, prompt_audio_codes) -> bool:
        request_rows = self._runtime.build_voice_clone_request_rows(
            prompt_audio_codes,
            self._runtime.encode_text(text),
        )
        if not self.streaming:
            generated_frames = self._runtime.generate_audio_frames(request_rows)
            waveform = self._runtime.decode_full_audio_safe(generated_frames)
            if not self.stop_synthesis_event.is_set():
                self._queue_audio(waveform)
            return True

        pending_decode_frames = []
        self._runtime.codec_streaming_session.reset()

        class _Stopped(Exception):
            pass

        def decode_pending(force: bool = False) -> None:
            if self.stop_synthesis_event.is_set():
                pending_decode_frames.clear()
                raise _Stopped()
            if not pending_decode_frames:
                return
            if not force and len(pending_decode_frames) < self.stream_decode_frames:
                return
            frame_budget = len(pending_decode_frames) if force else self.stream_decode_frames
            frame_chunk = pending_decode_frames[:frame_budget]
            del pending_decode_frames[:frame_budget]
            decoded = self._runtime.codec_streaming_session.run_frames(frame_chunk)
            if decoded is None:
                return
            audio, audio_length = decoded
            waveform = self._decode_onnx_audio(audio, int(audio_length))
            if waveform is not None:
                self._queue_audio(waveform)

        def on_frame(_generated_frames, _step_index, frame) -> None:
            if self.stop_synthesis_event.is_set():
                raise _Stopped()
            pending_decode_frames.append(list(frame))
            decode_pending(False)

        try:
            self._runtime.generate_audio_frames(request_rows, on_frame=on_frame)
            decode_pending(True)
        except _Stopped:
            return True
        finally:
            self._runtime.codec_streaming_session.reset()
        return True

    def _synthesize_onnx(self, text: str) -> bool:
        voice = self._voice()
        prepared = self._runtime.prepare_synthesis_text(
            text=text,
            voice=voice.voice or self.voice_name,
            prompt_text=voice.prompt_text or "",
            enable_wetext=self.enable_wetext_processing,
            enable_normalize_tts_text=self.enable_normalize_tts_text,
        )
        prepared_text = str(prepared["text"])
        prompt_audio_codes = self._onnx_prompt_audio_codes(voice)
        text_chunks = self._runtime.split_voice_clone_text(
            prepared_text,
            max_tokens=self.voice_clone_max_text_tokens,
        )
        for chunk_index, chunk_text in enumerate(text_chunks):
            if self.stop_synthesis_event.is_set():
                return True
            self._synthesize_onnx_text_chunk(chunk_text, prompt_audio_codes)
            if chunk_index < len(text_chunks) - 1:
                pause_seconds = self._runtime.estimate_voice_clone_inter_chunk_pause_seconds(chunk_text)
                pause = np.zeros(
                    (max(0, int(round(self.sampling_rate * pause_seconds))), self.channels),
                    dtype=np.float32,
                )
                self._queue_audio(pause)
        return True

    def _synthesize_torch(self, text: str) -> bool:
        voice = self._voice()
        output_path = self._output_path()
        if not self.streaming:
            result = self._runtime.synthesize(
                text=text,
                voice=voice.voice or self.voice_name,
                mode="voice_clone",
                output_audio_path=output_path,
                prompt_audio_path=voice.prompt_audio_path,
                prompt_text=None,
                max_new_frames=self.max_new_frames,
                voice_clone_max_text_tokens=self.voice_clone_max_text_tokens,
                voice_clone_max_memory_per_sample_gb=self.voice_clone_max_memory_per_sample_gb,
                tts_max_batch_size=self.tts_max_batch_size,
                codec_max_batch_size=self.codec_max_batch_size,
                do_sample=self.do_sample,
                text_temperature=self.text_temperature,
                text_top_p=self.text_top_p,
                text_top_k=self.text_top_k,
                audio_temperature=self.audio_temperature,
                audio_top_p=self.audio_top_p,
                audio_top_k=self.audio_top_k,
                audio_repetition_penalty=self.audio_repetition_penalty,
                seed=self.seed,
                attn_implementation=self.attn_implementation,
            )
            self.sampling_rate = int(result.get("sample_rate", self.sampling_rate))
            if not self.stop_synthesis_event.is_set():
                self._queue_audio(result["waveform_numpy"])
            return True

        for event in self._runtime.synthesize_stream(
            text=text,
            voice=voice.voice or self.voice_name,
            mode="voice_clone",
            output_audio_path=output_path,
            prompt_audio_path=voice.prompt_audio_path,
            prompt_text=None,
            max_new_frames=self.max_new_frames,
            voice_clone_max_text_tokens=self.voice_clone_max_text_tokens,
            voice_clone_max_memory_per_sample_gb=self.voice_clone_max_memory_per_sample_gb,
            tts_max_batch_size=self.tts_max_batch_size,
            codec_max_batch_size=self.codec_max_batch_size,
            do_sample=self.do_sample,
            text_temperature=self.text_temperature,
            text_top_p=self.text_top_p,
            text_top_k=self.text_top_k,
            audio_temperature=self.audio_temperature,
            audio_top_p=self.audio_top_p,
            audio_top_k=self.audio_top_k,
            audio_repetition_penalty=self.audio_repetition_penalty,
            seed=self.seed,
            attn_implementation=self.attn_implementation,
        ):
            if self.stop_synthesis_event.is_set():
                return True
            if event.get("type") != "audio":
                if event.get("type") == "result":
                    self.sampling_rate = int(event.get("sample_rate", self.sampling_rate))
                continue
            self.sampling_rate = int(event.get("sample_rate", self.sampling_rate))
            self._queue_audio(event["waveform_numpy"])
        return True

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)
        del sentence_count

        if self.stop_synthesis_event.is_set():
            return True

        try:
            if self.backend == "onnx":
                return self._synthesize_onnx(text)
            return self._synthesize_torch(text)
        except Exception:
            logging.exception("MOSS-TTS synthesis failed")
            return False

    def shutdown(self):
        self._runtime = None
        self._voices.clear()
        self._current_voice = None
        while self._dll_directory_handles:
            handle = self._dll_directory_handles.pop()
            close = getattr(handle, "close", None)
            if callable(close):
                close()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
