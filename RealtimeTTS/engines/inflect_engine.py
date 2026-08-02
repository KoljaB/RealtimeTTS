"""RealtimeTTS engine for the single-voice Inflect-Micro-v2 model."""

from __future__ import annotations

import _imp
import gc
import hashlib
import importlib.util
import logging
import os
import sys
import threading
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Union

import numpy as np

from .base_engine import BaseEngine


logger = logging.getLogger(__name__)

PYTORCH_MODEL_ID = "owensong/Inflect-Micro-v2"
PYTORCH_REVISION = "96e9360236ffcd734344067f20b4005b13e6358d"
ONNX_MODEL_ID = "owensong/Inflect-Micro-v2-ONNX"
ONNX_REVISION = "91b1ab6432323064ec0e8e9704d92fcecd24855f"
SAMPLE_RATE = 24_000

_DOWNLOAD_PATTERNS = {
    "pytorch": [
        "config.json",
        "model.pth",
        "inference.py",
        "inflect_nano_v2_frontend.py",
        "inflect_vits_frontend.py",
        "runtime/**",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "third_party/**",
    ],
    "onnx": [
        "onnx/duration.onnx",
        "onnx/decode.onnx",
        "onnx/inference_onnx.py",
        "inflect_nano_v2_frontend.py",
        "inflect_vits_frontend.py",
        "runtime/text/**",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "third_party/**",
    ],
}

_PINNED_HASHES = {
    "pytorch": {
        "config.json": "91ba9652f743f65e6be48eaa1aacea1db3d2941f5138ce309f359d4ce14f34e4",
        "model.pth": "3eede065c9ccfa88ade0a5a9a5c23de34afcbbb32213e59aad44d5cf100fdee8",
        "inference.py": "9bccc990aafa8e363a34269367a034fc69d8216f7317650be0e389bdd545eae3",
        "inflect_nano_v2_frontend.py": "399cba29408d594d655157ab6af925dc12edcd4207cf00595386106e7e53190a",
        "inflect_vits_frontend.py": "70eb4cdb96b2752015f72be984f5e62e0f26d7dc80e708875601702057b95cc2",
        "runtime/attentions.py": "dc65085d0e5d67da9f7603c98efad1e27eaf42c572f632a58080a5cc0f577f92",
        "runtime/commons.py": "f223dc1df791d2fb2953cfebd7014a7bc2a6ed614c05de6c17c7b44718a42cdd",
        "runtime/inflect_alias_free.py": "d2144153504a8986b737f2043468bf8bc73027ae8158cd1d8ca58ef54bf82ea7",
        "runtime/models.py": "f47a50c3381e7022dbd263c489f3c6b8f6c6c7ebb192729584af0ce0596b7aff",
        "runtime/modules.py": "f6514eaf7fab47d866ee0f9da94ca6ab6e45ebaf246151408da20dd14f242a75",
        "runtime/monotonic_align.py": "16ca4890f3e0107250b4a522c0127779994a5a2b99fc35d6a27f93496a72153a",
        "runtime/transforms.py": "9d95e518610c35581c4f1c3b4e8eeca91b7c3d6899e7f0ebb86a0387394d3046",
        "runtime/utils.py": "db11ba9b5ffd342abffc59229de7575a7d9e23468f91c6398e70eb4e01636716",
        "runtime/text/cleaners.py": "11dc7325547290e529c597e620f3aaf8d0bd8a2a2987d48eb42d2289534e42d1",
        "runtime/text/symbols.py": "e8871acbc37bd79f61ed063a2b3aa20e20555ddec15c6b9f29e1ab05e7f8b776",
        "runtime/text/__init__.py": "7f88b1aae463edcb42e1dce6f5c4227b167b3f5a40853bc644c73db015c1e471",
    },
    "onnx": {
        "onnx/decode.onnx": "7940923add86f76e7fa78d910b0632ca1779f8cc9a2ca2b49236381a9ca77183",
        "onnx/duration.onnx": "b728ca2564b9e5b7d6cf5e446f65e02a6fe2f1880ba281466fec93a667dd2388",
        "onnx/inference_onnx.py": (
            "cf48c4812f68e8315d80568e93650c64a7ad3fdfb9a0051c61a0d1d0746df274"
        ),
        "inflect_nano_v2_frontend.py": "399cba29408d594d655157ab6af925dc12edcd4207cf00595386106e7e53190a",
        "inflect_vits_frontend.py": "70eb4cdb96b2752015f72be984f5e62e0f26d7dc80e708875601702057b95cc2",
        "runtime/text/cleaners.py": "11dc7325547290e529c597e620f3aaf8d0bd8a2a2987d48eb42d2289534e42d1",
        "runtime/text/symbols.py": "e8871acbc37bd79f61ed063a2b3aa20e20555ddec15c6b9f29e1ab05e7f8b776",
        "runtime/text/__init__.py": "7f88b1aae463edcb42e1dce6f5c4227b167b3f5a40853bc644c73db015c1e471",
    },
}

_IMPORT_COLLISION_ROOTS = {
    "attentions",
    "commons",
    "inflect_alias_free",
    "inflect_nano_v2_frontend",
    "inflect_vits_frontend",
    "models",
    "modules",
    "monotonic_align",
    "text",
    "transforms",
    "utils",
}
_IMPORT_LOCK = threading.RLock()
_FRONTEND_LOCK = threading.Lock()
_SYNTHESIS_LOCK = threading.Lock()


@dataclass(frozen=True)
class InflectVoice:
    """The fixed synthetic male voice shipped with Inflect-Micro-v2."""

    name: str = "male"

    def __repr__(self) -> str:
        return self.name


def _is_collision_name(name: str) -> bool:
    return name.partition(".")[0] in _IMPORT_COLLISION_ROOTS


def _module_is_below(module: ModuleType, root: Path) -> bool:
    filename = getattr(module, "__file__", None)
    if not filename:
        return False
    try:
        # Keep the lexical snapshot path. Hugging Face model files may be
        # symlinks into a sibling blob cache, so resolving the file itself
        # would incorrectly make it appear to live outside model_root.
        module_path = Path(os.path.abspath(filename))
        root_path = Path(os.path.abspath(root))
        return module_path.is_relative_to(root_path)
    except (OSError, RuntimeError, ValueError):
        return False


def _load_upstream_module(
    entry_file: Path, model_root: Path
) -> tuple[ModuleType, dict[str, ModuleType]]:
    """Load Inflect's standalone runtime without leaking its generic imports."""

    unique_name = f"_realtimetts_inflect_{uuid.uuid4().hex}"
    with _IMPORT_LOCK:
        _imp.acquire_lock()
        original_path = list(sys.path)
        displaced: dict[str, ModuleType] = {}
        root_logger = logging.getLogger()
        original_root_level = root_logger.level
        logging_guard = logging.NullHandler()
        try:
            for name in list(sys.modules):
                if _is_collision_name(name):
                    displaced[name] = sys.modules.pop(name)

            spec = importlib.util.spec_from_file_location(unique_name, entry_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load Inflect runtime from {entry_file}")
            entry_module = importlib.util.module_from_spec(spec)
            sys.modules[unique_name] = entry_module

            # Hugging Face snapshots commonly expose source files as symlinks
            # into the blob cache.  Inflect's own bootstrap derives its
            # package directory from ``Path(__file__).resolve()``, which then
            # points at that cache rather than the snapshot and makes imports
            # such as ``commons`` fail.  Keep the lexical snapshot paths ahead
            # of any paths the upstream bootstrap adds so its runtime modules
            # remain importable in either a regular checkout or an HF snapshot.
            sys.path.insert(0, str(model_root / "runtime"))
            sys.path.insert(0, str(model_root))
            # The pinned upstream runtime calls logging.basicConfig() at import
            # time. A temporary root handler makes that call a no-op, keeping a
            # library import from installing a handler or enabling DEBUG logs.
            root_logger.addHandler(logging_guard)
            try:
                spec.loader.exec_module(entry_module)
            finally:
                root_logger.removeHandler(logging_guard)
                root_logger.setLevel(original_root_level)

            owned_modules = {
                name: module
                for name, module in sys.modules.items()
                if isinstance(module, ModuleType) and _module_is_below(module, model_root)
            }
            return entry_module, owned_modules
        finally:
            sys.path[:] = original_path
            for name, module in list(sys.modules.items()):
                if name == unique_name or (
                    isinstance(module, ModuleType) and _module_is_below(module, model_root)
                ):
                    sys.modules.pop(name, None)
            sys.modules.update(displaced)
            _imp.release_lock()


def _find_owned_module(modules: dict[str, ModuleType], filename: str) -> ModuleType:
    for module in modules.values():
        module_file = getattr(module, "__file__", "")
        if module_file and Path(module_file).name == filename:
            return module
    raise ImportError(f"Inflect runtime did not load required module {filename}")


def _isolate_upstream_logging(owned_modules: dict[str, ModuleType]) -> None:
    """Keep the training runtime's module-level logger away from the host root."""

    try:
        utils_module = _find_owned_module(owned_modules, "utils.py")
    except ImportError:
        return
    upstream_logger = logging.Logger(
        "RealtimeTTS.Inflect.upstream",
        level=logging.WARNING,
    )
    upstream_logger.addHandler(logging.NullHandler())
    utils_module.logger = upstream_logger


def _install_cached_frontend(
    entry_module: ModuleType,
    owned_modules: dict[str, ModuleType],
) -> tuple[Any, Any]:
    """Replace upstream's per-call eSpeak construction with one persistent backend."""

    nano_frontend = _find_owned_module(owned_modules, "inflect_nano_v2_frontend.py")
    vits_frontend = _find_owned_module(owned_modules, "inflect_vits_frontend.py")

    try:
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import default_separator
    except ImportError as exc:
        raise ImportError(
            "Inflect requires phonemizer and espeakng-loader. "
            "Install them with: pip install realtimetts[inflect]"
        ) from exc

    nano_frontend._configure_espeak()
    phonemizer_logger = logging.Logger(
        "RealtimeTTS.Inflect.phonemizer", level=logging.ERROR
    )
    espeak_backend = EspeakBackend(
        language="en-us",
        preserve_punctuation=True,
        with_stress=True,
        logger=phonemizer_logger,
    )

    def cached_frontend(text: str) -> Any:
        with _FRONTEND_LOCK:
            normalized = nano_frontend.normalize_text(text)
            phonemes = espeak_backend.phonemize(
                [normalized],
                separator=default_separator,
                strip=True,
                njobs=1,
            )[0]
        return vits_frontend.VitsFrontendOutput(
            raw_text=text,
            normalized_text=normalized,
            phoneme_text=vits_frontend._apply_phoneme_overrides(phonemes),
        )

    entry_module.run_vits_frontend = cached_frontend
    return espeak_backend, cached_frontend


class InflectEngine(BaseEngine):
    """Low-latency RealtimeTTS wrapper for Inflect-Micro-v2.

    ``backend="auto"`` selects optimized PyTorch when CUDA is available, ONNX
    on CPU when it is installed, and PyTorch CPU otherwise.
    """

    def __init__(
        self,
        voice: Optional[Union[str, InflectVoice]] = None,
        *,
        backend: str = "auto",
        device: str = "auto",
        model_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        token: Optional[str] = None,
        local_files_only: bool = False,
        speed: float = 1.0,
        variation: float = 0.667,
        seed: int = 0,
        cpu_threads: Optional[int] = None,
        warmup: bool = True,
        warmup_text: str = "Ready.",
        verify_files: bool = True,
        debug: bool = False,
    ) -> None:
        self.debug = bool(debug)
        # PyTorch's upstream runtime seeds process-global RNG state, so calls
        # must be serialized across engine instances, not only per instance.
        self._synthesis_lock = _SYNTHESIS_LOCK
        self._runtime: Any = None
        self._upstream_entry: Optional[ModuleType] = None
        self._upstream_modules: dict[str, ModuleType] = {}
        self._espeak_backend: Any = None
        self._cached_frontend: Any = None

        self.backend, self.device = self._resolve_backend_and_device(backend, device)
        self.model_id, self.revision = self._default_source()
        self.cache_dir = str(Path(cache_dir).expanduser()) if cache_dir is not None else None
        self.token = token
        self.local_files_only = bool(local_files_only)
        self.verify_files = bool(verify_files)

        self.speed = self._validate_speed(speed)
        self.variation = self._validate_variation(variation)
        self.seed = self._validate_seed(seed)
        self.cpu_threads = self._validate_cpu_threads(cpu_threads)
        self.voice = InflectVoice()
        if voice is not None:
            self.set_voice(voice)

        self.model_dir = self._resolve_model_dir(model_dir)
        self._validate_runtime_files()
        normalized_warmup = " ".join(warmup_text.split()) if warmup else ""
        if warmup:
            if not normalized_warmup:
                raise ValueError("warmup_text must contain speakable text when warmup=True")

        with self._synthesis_lock:
            with self._torch_rng_scope():
                self._runtime = self._load_model()
                self.sample_rate = int(
                    getattr(self._runtime, "sample_rate", SAMPLE_RATE)
                )
                if warmup:
                    sample_rate, _ = self._runtime.synthesize(
                        normalized_warmup,
                        speed=self.speed,
                        variation=self.variation,
                        seed=self.seed,
                    )
                    self.sample_rate = int(sample_rate)

    def post_init(self) -> None:
        self.engine_name = "inflect"
        self.preload_sentence_tokenizer = True

    @staticmethod
    def _validate_speed(value: Real) -> float:
        if not isinstance(value, Real) or not 0.5 <= float(value) <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        return float(value)

    @staticmethod
    def _validate_variation(value: Real) -> float:
        if not isinstance(value, Real) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("variation must be between 0.0 and 1.0")
        return float(value)

    @staticmethod
    def _validate_seed(value: Integral) -> int:
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError("seed must be a non-negative integer")
        if not 0 <= int(value) <= 2**63 - 1:
            raise ValueError("seed must be between 0 and 2^63 - 1")
        return int(value)

    @staticmethod
    def _validate_cpu_threads(value: Optional[Integral]) -> int:
        if value is None:
            return min(8, os.cpu_count() or 1)
        if not isinstance(value, Integral) or int(value) < 1:
            raise ValueError("cpu_threads must be a positive integer")
        return int(value)

    @staticmethod
    def _resolve_backend_and_device(backend: str, device: str) -> tuple[str, str]:
        requested_backend = str(backend).lower()
        requested_device = str(device).lower()
        if requested_backend not in {"auto", "pytorch", "onnx"}:
            raise ValueError("backend must be 'auto', 'pytorch', or 'onnx'")

        if requested_backend == "auto":
            if requested_device == "auto":
                try:
                    import torch
                except ImportError:
                    torch = None

                try:
                    onnx_available = (
                        importlib.util.find_spec("onnxruntime") is not None
                    )
                except (ImportError, ValueError):
                    onnx_available = False
                try:
                    scipy_available = importlib.util.find_spec("scipy") is not None
                except (ImportError, ValueError):
                    scipy_available = False

                if (
                    torch is not None
                    and torch.cuda.is_available()
                    and (scipy_available or not onnx_available)
                ):
                    requested_backend, requested_device = "pytorch", "cuda:0"
                elif onnx_available:
                    requested_backend, requested_device = "onnx", "cpu"
                elif torch is not None:
                    requested_backend, requested_device = "pytorch", "cpu"
                else:
                    requested_backend, requested_device = "onnx", "cpu"
            elif requested_device == "cpu" or requested_device == "directml":
                requested_backend = "onnx"
            else:
                requested_backend = "pytorch"

        if requested_device == "auto":
            if requested_backend == "onnx":
                requested_device = "cpu"
            else:
                import torch

                requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return requested_backend, requested_device

    def _default_source(self) -> tuple[str, str]:
        if self.backend == "onnx":
            return ONNX_MODEL_ID, ONNX_REVISION
        return PYTORCH_MODEL_ID, PYTORCH_REVISION

    def _torch_rng_scope(self) -> Any:
        if self.backend != "pytorch":
            return nullcontext()
        import torch

        devices = (
            list(range(torch.cuda.device_count()))
            if self.device.startswith("cuda")
            else []
        )
        return torch.random.fork_rng(devices=devices, enabled=True)

    def _resolve_model_dir(self, model_dir: Optional[Union[str, Path]]) -> Path:
        if model_dir is not None:
            path = Path(model_dir).expanduser().resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"Inflect model directory does not exist: {path}")
            return path

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Inflect model download requires huggingface-hub. "
                "Install it with: pip install realtimetts[inflect]"
            ) from exc

        kwargs: dict[str, Any] = {
            "repo_id": self.model_id,
            "allow_patterns": _DOWNLOAD_PATTERNS[self.backend],
            "local_files_only": self.local_files_only,
        }
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        if self.token is not None:
            kwargs["token"] = self.token
        return Path(snapshot_download(**kwargs)).resolve()

    def _entry_file(self) -> Path:
        if self.backend == "onnx":
            return self.model_dir / "onnx" / "inference_onnx.py"
        return self.model_dir / "inference.py"

    def _validate_runtime_files(self) -> None:
        required = list(_PINNED_HASHES[self.backend])
        missing = [relative for relative in required if not (self.model_dir / relative).is_file()]
        if missing:
            missing_files = ", ".join(missing)
            raise FileNotFoundError(
                f"Inflect {self.backend} model directory is incomplete; "
                f"missing: {missing_files}"
            )

        if not self.verify_files:
            return
        for relative, expected in _PINNED_HASHES[self.backend].items():
            digest = hashlib.sha256()
            with (self.model_dir / relative).open("rb") as file:
                for block in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(block)
            actual = digest.hexdigest()
            if actual != expected:
                raise RuntimeError(
                    f"Inflect file checksum mismatch for {relative}: "
                    f"expected {expected}, got {actual}"
                )

    def _load_model(self) -> Any:
        try:
            entry_module, owned_modules = _load_upstream_module(
                self._entry_file(), self.model_dir
            )
            _isolate_upstream_logging(owned_modules)
            espeak_backend, cached_frontend = _install_cached_frontend(
                entry_module, owned_modules
            )
            if self.backend == "onnx":
                runtime = self._create_onnx_runtime(entry_module)
            else:
                runtime = self._create_pytorch_runtime(entry_module)
        except ImportError as exc:
            raise ImportError(
                "Could not load Inflect-Micro-v2. Install its dependencies with: "
                "pip install realtimetts[inflect]"
            ) from exc

        self._upstream_entry = entry_module
        self._upstream_modules = owned_modules
        self._espeak_backend = espeak_backend
        self._cached_frontend = cached_frontend
        return runtime

    def _create_pytorch_runtime(self, entry_module: ModuleType) -> Any:
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device!r} was requested, but CUDA is unavailable in PyTorch"
            )
        return entry_module.InflectTTS(self.model_dir, self.device)

    def _create_onnx_runtime(self, entry_module: ModuleType) -> Any:
        ort = entry_module.ort
        provider_name = "cuda" if self.device.startswith("cuda") else self.device
        selected_provider = entry_module.available_provider(provider_name)
        providers = [selected_provider]
        if selected_provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.cpu_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        runtime = entry_module.InflectONNX.__new__(entry_module.InflectONNX)
        onnx_dir = self.model_dir / "onnx"
        runtime.duration = ort.InferenceSession(
            str(onnx_dir / "duration.onnx"),
            sess_options=options,
            providers=providers,
        )
        runtime.decode = ort.InferenceSession(
            str(onnx_dir / "decode.onnx"),
            sess_options=options,
            providers=providers,
        )
        return runtime

    def get_stream_info(self) -> tuple[int, int, int]:
        import pyaudio

        return pyaudio.paInt16, 1, self.sample_rate

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        del sentence_count
        with self._synthesis_lock:
            super().synthesize(text)
            normalized = " ".join(text.split())
            if not normalized:
                return True
            if self.stop_synthesis_event.is_set():
                return True
            if self._runtime is None:
                logger.error("Inflect engine has already been shut down")
                return False

            try:
                with self._torch_rng_scope():
                    sample_rate, waveform = self._runtime.synthesize(
                        normalized,
                        speed=self.speed,
                        variation=self.variation,
                        seed=self.seed,
                    )
                if self.stop_synthesis_event.is_set():
                    return True

                audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
                if audio.size == 0:
                    logger.error("Inflect returned an empty waveform")
                    return False
                sample_rate = int(sample_rate)
                if sample_rate != self.sample_rate:
                    raise RuntimeError(
                        f"Inflect sample rate changed from {self.sample_rate} to {sample_rate} Hz"
                    )
                pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                self.audio_duration += audio.size / sample_rate
                self.queue.put(pcm.tobytes())
                return True
            except Exception as exc:
                if self.debug:
                    logger.exception("Inflect synthesis failed")
                else:
                    logger.error("Inflect synthesis failed: %s", exc)
                return False

    def get_voices(self) -> list[InflectVoice]:
        return [self.voice]

    def set_voice(self, voice: Union[str, InflectVoice]) -> None:
        name = voice.name if isinstance(voice, InflectVoice) else str(voice)
        if name.strip().lower() not in {
            "male",
            "default",
            "inflect",
            "inflect-micro-v2",
        }:
            raise ValueError("Inflect-Micro-v2 has one fixed voice: 'male'")
        self.voice = InflectVoice()

    def set_voice_parameters(self, **voice_parameters: Any) -> None:
        supported = {"speed", "variation", "seed"}
        unknown = set(voice_parameters) - supported
        if unknown:
            unsupported = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported Inflect voice parameter(s): {unsupported}")
        speed = (
            self._validate_speed(voice_parameters["speed"])
            if "speed" in voice_parameters
            else self.speed
        )
        variation = (
            self._validate_variation(voice_parameters["variation"])
            if "variation" in voice_parameters
            else self.variation
        )
        seed = (
            self._validate_seed(voice_parameters["seed"])
            if "seed" in voice_parameters
            else self.seed
        )
        with self._synthesis_lock:
            self.speed = speed
            self.variation = variation
            self.seed = seed

    def shutdown(self) -> None:
        with self._synthesis_lock:
            self._runtime = None
            self._upstream_entry = None
            self._upstream_modules = {}
            self._espeak_backend = None
            self._cached_frontend = None
            gc.collect()
            if self.backend == "pytorch" and self.device.startswith("cuda"):
                try:
                    import torch

                    torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    pass


__all__ = ["InflectEngine", "InflectVoice"]
