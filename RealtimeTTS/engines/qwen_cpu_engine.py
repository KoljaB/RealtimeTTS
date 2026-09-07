"""Qwen voice cloning through the CPU-only native runtime."""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from .qwen_engine import DEFAULT_MODEL, QwenEngine, QwenEngineError, QwenVoice


class QwenCpuEngine(QwenEngine):
    """CPU Qwen synthesis with the same voices, streaming and controls as QwenEngine.

    Requires the CPU native wheel. The binding checks the loaded library before
    creating a context, including when ``library_path`` is supplied. This engine
    never changes process-wide GPU visibility or backend environment variables.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        cpu_threads: Optional[int] = None,
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
        self.cpu_threads = cpu_threads
        self.device = "cpu"
        self.cpu_only = True
        kwargs.setdefault("clamp_fp16", False)
        kwargs.setdefault("clone_mode", "speaker_only")
        super().__init__(model_id, **kwargs)

    def post_init(self) -> None:
        self.engine_name = "qwen_cpu"

    def _backend_options(self) -> dict[str, Any]:
        return {"cpu_threads": self.cpu_threads}

    def _cache_identity(self, voice: QwenVoice) -> dict[str, Any]:
        return {**super()._cache_identity(voice), "device": "cpu"}

    def _create_backend(
        self, backend_factory: Optional[Callable[..., Any]]
    ) -> tuple[Any, int, str]:
        if backend_factory is None:
            try:
                import qwentts_cpp
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
            "QwenCpuEngine requires the realtimetts-qwen-native CPU wheel "
            "(0.2.0+cpu1). Install realtimetts[qwen-cpu] or "
            "realtimetts[qwen-cpu-server] from the CPU wheelhouse."
        )
