"""Qwen voice cloning through the CPU-only native runtime."""

from __future__ import annotations

import importlib
import os
import time
from typing import Any, Callable, Optional

import numpy as np

from .qwen_engine import (
    DEFAULT_MODEL,
    MODEL_TYPE_BASE,
    QwenEngine,
    QwenEngineError,
    QwenVoice,
    SAMPLE_RATE,
)


CPU_ONSET_BASELINE_PROFILE = "qwen3_tts_12hz_0_6b_base_q8_v1"
CPU_ONSET_RECOVERY_PROFILE = "qwen3_tts_12hz_0_6b_base_q8_cpu_recovery_v2"
CPU_ONSET_RECOVERY_WINDOW_MS = 240.0
CPU_ONSET_RECOVERY_WINDOW_SAMPLES = int(
    SAMPLE_RATE * CPU_ONSET_RECOVERY_WINDOW_MS / 1000.0
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
        onset_silence_recovery: bool = False,
        **kwargs: Any,
    ) -> None:
        self.onset_silence_recovery = bool(onset_silence_recovery)
        self.onset_silence_recovery_profile = CPU_ONSET_RECOVERY_PROFILE
        self._onset_recovery_state: Optional[dict[str, Any]] = None
        self._onset_recovery_first_profile: Optional[dict[str, Any]] = None
        if self.onset_silence_recovery:
            configured_profile = str(
                kwargs.get("onset_silence_profile", "off") or "off"
            ).strip().lower()
            if configured_profile != CPU_ONSET_BASELINE_PROFILE:
                raise ValueError(
                    "onset_silence_recovery requires "
                    f"onset_silence_profile={CPU_ONSET_BASELINE_PROFILE!r}"
                )
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

    def _validate_additional_onset_silence_profiles(self) -> None:
        """Validate the recovery profile while the native context is loading."""

        if not self.onset_silence_recovery:
            return
        validator = getattr(self._backend, "validate_onset_silence_profile", None)
        if not callable(validator):
            raise QwenEngineError(
                "onset_silence_recovery requires a native binding that validates "
                "the CPU recovery onset profile"
            )
        normalized = str(validator(CPU_ONSET_RECOVERY_PROFILE)).strip().lower()
        if normalized != CPU_ONSET_RECOVERY_PROFILE:
            raise QwenEngineError(
                "The native binding resolved the CPU onset recovery profile to "
                f"{normalized!r}, expected {CPU_ONSET_RECOVERY_PROFILE!r}"
            )
        self.onset_silence_recovery_profile = normalized

    def _recovery_elapsed_ms(self, stream_started_ns: int) -> float:
        return (time.perf_counter_ns() - stream_started_ns) / 1_000_000.0

    def _new_onset_recovery_state(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "window_ms": CPU_ONSET_RECOVERY_WINDOW_MS,
            "threshold": self.silence_threshold,
            "triggered": False,
            "trigger_ms": None,
            "aborted": False,
            "abort_ms": None,
            "abort_reason": None,
            "retry_attempted": False,
            "retry_started": False,
            "retry_attempt": None,
        }

    def _mark_onset_abort(
        self,
        state: dict[str, Any],
        reason: str,
        stream_started_ns: int,
    ) -> None:
        state["aborted"] = True
        state["abort_reason"] = reason
        if state["abort_ms"] is None:
            state["abort_ms"] = self._recovery_elapsed_ms(stream_started_ns)

    @staticmethod
    def _cancel_event_cancelled(cancel_event: Any) -> bool:
        """Check cancellation without acknowledging a native pause checkpoint."""

        if cancel_event is None:
            return False
        cancelled = getattr(cancel_event, "cancelled", None)
        if callable(cancelled):
            return bool(cancelled())
        is_set = getattr(cancel_event, "is_set", None)
        return bool(is_set()) if callable(is_set) else False

    def _probe_has_speech(self, probe: np.ndarray) -> bool:
        window_samples = self._silence_trim_window_samples(SAMPLE_RATE)
        if probe.size == 0:
            return False
        for start in range(0, int(probe.size), window_samples):
            end = start + window_samples
            if self._window_has_speech(probe[start:end], self.silence_threshold):
                return True
        return False

    @staticmethod
    def _window_has_speech(window: np.ndarray, threshold: float) -> bool:
        return bool(window.size and float(np.mean(np.abs(window))) > threshold)

    def _close_native_stream(self, stream: Any, *, require_join: bool = False) -> None:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
        if not require_join:
            return
        profile = dict(getattr(self._backend, "last_stream_profile", None) or {})
        producer_alive = profile.get("producer_alive_after_close")
        if producer_alive is None:
            raise QwenEngineError(
                "Qwen CPU onset recovery requires the native stream to report "
                "producer_alive_after_close after close()"
            )
        if bool(producer_alive):
            raise QwenEngineError(
                "Qwen CPU onset recovery aborted because the native stream "
                "producer remained alive after close()"
            )

    def _stream_native(
        self,
        stream_kwargs: dict[str, Any],
        *,
        stream_started_ns: Optional[int] = None,
    ) -> Any:
        """Wrap one CPU stream with one bounded quiet-onset retry."""

        if not self.onset_silence_recovery:
            self._onset_recovery_state = None
            self._onset_recovery_first_profile = None
            return super()._stream_native(
                stream_kwargs, stream_started_ns=stream_started_ns
            )

        started_ns = stream_started_ns or time.perf_counter_ns()
        state = self._new_onset_recovery_state()
        self._onset_recovery_state = state
        self._onset_recovery_first_profile = None
        if not self.trim_silence:
            self._mark_onset_abort(state, "trim_silence_disabled", started_ns)
            return super()._stream_native(
                stream_kwargs, stream_started_ns=stream_started_ns
            )
        if stream_kwargs.get("ref_codes") is not None or stream_kwargs.get("ref_text"):
            self._mark_onset_abort(state, "icl_voice", started_ns)
            return super()._stream_native(
                stream_kwargs, stream_started_ns=stream_started_ns
            )
        if self.model_type != MODEL_TYPE_BASE:
            self._mark_onset_abort(state, "non_base_model", started_ns)
            return super()._stream_native(
                stream_kwargs, stream_started_ns=stream_started_ns
            )
        if stream_kwargs.get("onset_silence_profile") != CPU_ONSET_BASELINE_PROFILE:
            self._mark_onset_abort(state, "baseline_profile_not_active", started_ns)
            return super()._stream_native(
                stream_kwargs, stream_started_ns=stream_started_ns
            )
        return self._stream_with_onset_recovery(
            stream_kwargs, started_ns, state
        )

    def _stream_with_onset_recovery(
        self,
        stream_kwargs: dict[str, Any],
        stream_started_ns: int,
        state: dict[str, Any],
    ) -> Any:
        """Run the first stream and replace it once after a quiet 240 ms."""

        cancel_event = stream_kwargs.get("cancel_event")
        first_stream = super()._stream_native(
            stream_kwargs, stream_started_ns=stream_started_ns
        )
        active_stream: Any = first_stream
        probe = np.empty(0, dtype=np.float32)
        probe_samples = 0
        speech_seen = False
        try:
            for chunk, sample_rate in first_stream:
                if self.stop_synthesis_event.is_set() or (
                    self._cancel_event_cancelled(cancel_event)
                ):
                    self._mark_onset_abort(state, "cancelled", stream_started_ns)
                    return
                if int(sample_rate) != SAMPLE_RATE:
                    self._mark_onset_abort(state, "invalid_sample_rate", stream_started_ns)
                    yield chunk, sample_rate
                    continue

                audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
                if (
                    not speech_seen
                    and not state["aborted"]
                    and probe_samples < CPU_ONSET_RECOVERY_WINDOW_SAMPLES
                ):
                    needed = CPU_ONSET_RECOVERY_WINDOW_SAMPLES - probe_samples
                    take = min(int(audio.size), needed)
                    if take:
                        probe_piece = np.nan_to_num(
                            audio[:take], nan=0.0, posinf=1.0, neginf=-1.0
                        )
                        probe = np.concatenate((probe, probe_piece))
                        probe_samples += take
                        if self._probe_has_speech(probe):
                            speech_seen = True
                            self._mark_onset_abort(
                                state, "speech_detected", stream_started_ns
                            )
                            yield chunk, sample_rate
                            continue
                        if probe_samples >= CPU_ONSET_RECOVERY_WINDOW_SAMPLES:
                            crossing_audio = np.nan_to_num(
                                audio[take:], nan=0.0, posinf=1.0, neginf=-1.0
                            )
                            if crossing_audio.size and self._probe_has_speech(
                                np.concatenate((probe, crossing_audio))
                            ):
                                speech_seen = True
                                self._mark_onset_abort(
                                    state, "speech_detected", stream_started_ns
                                )
                                yield chunk, sample_rate
                                continue
                            # Serialize the pause check with the close. This
                            # prevents a pause request from racing between the
                            # check and close(), where the native producer could
                            # otherwise be blocked in its pause checkpoint.
                            with self._control_condition:
                                if self._pause_requested:
                                    paused_before_retry = True
                                else:
                                    paused_before_retry = False
                                    state["triggered"] = True
                                    state["trigger_ms"] = self._recovery_elapsed_ms(
                                        stream_started_ns
                                    )
                                    state["aborted"] = True
                                    state["abort_reason"] = "quiet_window"
                                    # Do not set the shared request cancellation
                                    # event. The native generator owns its
                                    # internal cancellation and close() joins
                                    # its producer before retrying.
                                    self._close_native_stream(
                                        first_stream, require_join=True
                                    )
                                    active_stream = None
                            if paused_before_retry:
                                self._mark_onset_abort(
                                    state, "paused_before_retry", stream_started_ns
                                )
                                # Keep the original stream alive while paused;
                                # closing it here could join a producer blocked
                                # in its native pause checkpoint.
                                yield chunk, sample_rate
                                continue
                            state["abort_ms"] = self._recovery_elapsed_ms(
                                stream_started_ns
                            )
                            self._onset_recovery_first_profile = dict(
                                getattr(self._backend, "last_stream_profile", None)
                                or {}
                            )
                            if self.stop_synthesis_event.is_set() or (
                                self._cancel_event_cancelled(cancel_event)
                            ):
                                state["abort_reason"] = "cancelled_before_retry"
                                return
                            if audio.size > take:
                                yield np.asarray(
                                    audio[:take], dtype=np.float32
                                ), sample_rate
                            else:
                                yield chunk, sample_rate

                            if self.stop_synthesis_event.is_set() or (
                                self._cancel_event_cancelled(cancel_event)
                            ):
                                state["abort_reason"] = "cancelled_before_retry"
                                return

                            state["retry_attempted"] = True
                            if self.stop_synthesis_event.is_set() or (
                                self._cancel_event_cancelled(cancel_event)
                            ):
                                state["abort_reason"] = "cancelled_before_retry"
                                return
                            retry_kwargs = dict(stream_kwargs)
                            retry_kwargs[
                                "onset_silence_profile"
                            ] = self.onset_silence_recovery_profile
                            try:
                                retry_stream = super()._stream_native(
                                    retry_kwargs,
                                    stream_started_ns=stream_started_ns,
                                )
                            except BaseException:
                                state["abort_reason"] = "retry_start_failed"
                                state["retry_attempt"] = dict(
                                    getattr(self._backend, "last_stream_profile", None)
                                    or {}
                                )
                                raise
                            state["retry_started"] = True
                            active_stream = retry_stream
                            try:
                                for retry_chunk in retry_stream:
                                    yield retry_chunk
                            finally:
                                try:
                                    self._close_native_stream(
                                        retry_stream, require_join=True
                                    )
                                finally:
                                    state["retry_attempt"] = dict(
                                        getattr(self._backend, "last_stream_profile", None)
                                        or {}
                                    )
                                    active_stream = None
                            return

                yield chunk, sample_rate

            if state["abort_reason"] is None:
                if self.stop_synthesis_event.is_set() or (
                    self._cancel_event_cancelled(cancel_event)
                ):
                    self._mark_onset_abort(state, "cancelled", stream_started_ns)
                else:
                    state["abort_reason"] = "completed_before_window"
        except BaseException:
            if (
                self.stop_synthesis_event.is_set()
                or self._cancel_event_cancelled(cancel_event)
            ) and not state["retry_started"]:
                self._mark_onset_abort(state, "cancelled", stream_started_ns)
            elif state["abort_reason"] is None:
                state["abort_reason"] = "first_attempt_failed"
            raise
        finally:
            if active_stream is not None:
                self._close_native_stream(active_stream)
            if self._onset_recovery_first_profile is None:
                self._onset_recovery_first_profile = dict(
                    getattr(self._backend, "last_stream_profile", None) or {}
                )
            if state["retry_started"] and state["retry_attempt"] is None:
                state["retry_attempt"] = dict(
                    getattr(self._backend, "last_stream_profile", None) or {}
                )

    def _native_stream_profile(self) -> dict[str, Any]:
        state = self._onset_recovery_state
        if state is not None and state["triggered"]:
            return dict(self._onset_recovery_first_profile or {})
        return dict(getattr(self._backend, "last_stream_profile", None) or {})

    def _native_stream_recovery_profile(self) -> Optional[dict[str, Any]]:
        state = self._onset_recovery_state
        if state is None:
            return None
        first_attempt = self._onset_recovery_first_profile
        if first_attempt is None:
            first_attempt = dict(
                getattr(self._backend, "last_stream_profile", None) or {}
            )
        return {
            "mode": "cpu_onset_silence_recovery",
            "enabled": bool(state["enabled"]),
            "window_ms": float(state["window_ms"]),
            "threshold": float(state["threshold"]),
            "triggered": bool(state["triggered"]),
            "trigger_ms": state["trigger_ms"],
            "aborted": bool(state["aborted"]),
            "abort_ms": state["abort_ms"],
            "abort_reason": state["abort_reason"],
            "retry_attempted": bool(state["retry_attempted"]),
            "retry_started": bool(state["retry_started"]),
            "first_attempt": dict(first_attempt),
            "retry_attempt": (
                dict(state["retry_attempt"])
                if state["retry_attempt"] is not None
                else None
            ),
        }

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
