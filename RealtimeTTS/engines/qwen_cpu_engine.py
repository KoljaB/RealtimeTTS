"""Qwen voice cloning through the CPU-only native runtime."""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Optional

from .qwen_engine import (
    DEFAULT_MODEL,
    QwenEngine,
    QwenEngineError,
    QwenVoice,
)


class QwenCpuEngine(QwenEngine):
    """CPU Qwen synthesis with the same voices, streaming and controls as QwenEngine.

    Requires the CPU native wheel. The binding checks the loaded library before
    creating a context, including when ``library_path`` is supplied. This engine
    never changes process-wide GPU visibility or backend environment variables.

    ``cpu_codec_threads > 0`` overlaps streaming codec decoding with generation.
    ``cpu_stream_frames=2`` delivers 160 ms steady-state chunks (default: 4/320 ms).
    Optional affinity masks assign the two pools to logical CPUs without changing
    process affinity. These scheduling options require a native wheel exposing
    ``qt_init_cpu_ex``; they leave weights, precision and sampling unchanged.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        cpu_threads: Optional[int] = None,
        cpu_codec_threads: int = 0,
        cpu_stream_frames: int = 0,
        cpu_affinity: int = 0,
        cpu_codec_affinity: int = 0,
        **kwargs: Any,
    ) -> None:
        max_threads = min(256, os.cpu_count() or 1)
        if cpu_threads is None:
            cpu_threads = min(8, max_threads)
        if (
            isinstance(cpu_threads, bool)
            or not isinstance(cpu_threads, int)
            or not 1 <= cpu_threads <= max_threads
        ):
            raise ValueError(f"cpu_threads must be an integer between 1 and {max_threads}")
        for name, value in (("cpu_codec_threads", cpu_codec_threads),
                            ("cpu_stream_frames", cpu_stream_frames),
                            ("cpu_affinity", cpu_affinity),
                            ("cpu_codec_affinity", cpu_codec_affinity)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 0 <= cpu_codec_threads <= max_threads:
            raise ValueError(f"cpu_codec_threads must be between 0 and {max_threads}")
        if cpu_stream_frames not in (0, 1, 2, 4):
            raise ValueError("cpu_stream_frames must be 0, 1, 2 or 4")
        for name, mask, threads in (("cpu_affinity", cpu_affinity, cpu_threads),
                                   ("cpu_codec_affinity", cpu_codec_affinity, cpu_codec_threads)):
            if not 0 <= mask < 2**64:
                raise ValueError(f"{name} must be an unsigned 64-bit CPU mask")
            if mask and mask.bit_count() < threads:
                raise ValueError(f"{name} needs at least {threads} CPUs")
        if cpu_codec_affinity and not cpu_codec_threads:
            raise ValueError("cpu_codec_affinity requires cpu_codec_threads")
        self.cpu_threads = cpu_threads
        self.cpu_codec_threads = cpu_codec_threads
        self.cpu_stream_frames = cpu_stream_frames
        self.cpu_affinity = cpu_affinity
        self.cpu_codec_affinity = cpu_codec_affinity
        self.device = "cpu"
        self.cpu_only = True
        kwargs.setdefault("clamp_fp16", False)
        kwargs.setdefault("clone_mode", "speaker_only")
        super().__init__(model_id, **kwargs)

    def post_init(self) -> None:
        self.engine_name = "qwen_cpu"

    def _backend_options(self) -> dict[str, Any]:
        options = {"cpu_threads": self.cpu_threads}
        # Preserve support for existing wheels when scheduling is not requested.
        for name in ("cpu_codec_threads", "cpu_stream_frames", "cpu_affinity", "cpu_codec_affinity"):
            value = getattr(self, name)
            if value:
                options[name] = value
        return options

    def _cache_identity(self, voice: QwenVoice) -> dict[str, Any]:
        return {**super()._cache_identity(voice), "device": "cpu"}

    def _load_native_module(self):
        try:
            return importlib.import_module("qwentts_cpp_cpu")
        except ModuleNotFoundError as exc:
            if exc.name != "qwentts_cpp_cpu":
                raise
            # Existing attested private CPU deployments used qwentts_cpp.
            return super()._load_native_module()

    def _create_backend(
        self, backend_factory: Optional[Callable[..., Any]]
    ) -> tuple[Any, int, str]:
        if backend_factory is None:
            try:
                qwentts_cpp = self._load_native_module()
            except ImportError as exc:
                raise QwenEngineError(self._installation_help()) from exc
            if getattr(qwentts_cpp, "CPU_ONLY", False) is not True:
                raise QwenEngineError(self._installation_help())
        return super()._create_backend(backend_factory)

    def _translate_error(self, exc: BaseException, action: str) -> BaseException:
        if isinstance(exc, QwenEngineError):
            return exc
        if "out of memory" in str(exc).lower():
            return QwenEngineError(
                f"QwenCpuEngine failed while {action}: insufficient system RAM. {exc}"
            )
        return super()._translate_error(exc, action)

    @staticmethod
    def _installation_help() -> str:
        return (
            "QwenCpuEngine requires the CPU wheel realtimetts-qwen-native-cpu "
            "(0.3.0). Install realtimetts[qwen-cpu] or "
            "realtimetts[qwen-cpu-server]."
        )
