import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from RealtimeTTS.engines.inflect_engine import (
    InflectEngine,
    InflectVoice,
    ONNX_MODEL_ID,
    ONNX_REVISION,
    _PINNED_HASHES,
    _isolate_upstream_logging,
    _load_upstream_module,
    _module_is_below,
)


class FakeInflectRuntime:
    sample_rate = 24_000

    def __init__(self, waveform=None):
        self.waveform = np.asarray(
            waveform if waveform is not None else [-1.0, -0.5, 0.0, 0.5, 1.0],
            dtype=np.float32,
        )
        self.calls = []
        self.on_synthesize = None

    def synthesize(self, text, *, speed, variation, seed):
        self.calls.append(
            {
                "text": text,
                "speed": speed,
                "variation": variation,
                "seed": seed,
            }
        )
        if self.on_synthesize is not None:
            self.on_synthesize()
        return self.sample_rate, self.waveform.copy()


@pytest.fixture
def fake_runtime():
    return FakeInflectRuntime()


@pytest.fixture
def engine_factory(monkeypatch, tmp_path, fake_runtime):
    monkeypatch.setattr(InflectEngine, "_validate_runtime_files", lambda self: None)
    monkeypatch.setattr(InflectEngine, "_load_model", lambda self: fake_runtime)

    def create(**kwargs):
        options = {
            "backend": "onnx",
            "device": "cpu",
            "model_dir": tmp_path,
            "warmup": False,
            "verify_files": False,
        }
        options.update(kwargs)
        return InflectEngine(**options)

    return create


def test_public_lazy_exports():
    from RealtimeTTS import InflectEngine as RootEngine
    from RealtimeTTS import InflectVoice as RootVoice
    from RealtimeTTS.engines import InflectEngine as EnginesEngine
    from RealtimeTTS.engines import InflectVoice as EnginesVoice

    assert RootEngine is InflectEngine
    assert RootVoice is InflectVoice
    assert EnginesEngine is InflectEngine
    assert EnginesVoice is InflectVoice


def test_upstream_loader_restores_colliding_modules_and_sys_path(
    monkeypatch, tmp_path
):
    (tmp_path / "utils.py").write_text("VALUE = 'inflect'\n", encoding="utf-8")
    entry_file = tmp_path / "inference.py"
    entry_file.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent))\n"
        "import utils\n"
        "VALUE = utils.VALUE\n",
        encoding="utf-8",
    )
    sentinel = ModuleType("utils")
    sentinel.VALUE = "original"
    monkeypatch.setitem(sys.modules, "utils", sentinel)
    original_path = list(sys.path)

    entry_module, owned_modules = _load_upstream_module(entry_file, tmp_path)

    assert entry_module.VALUE == "inflect"
    assert owned_modules["utils"].VALUE == "inflect"
    assert sys.modules["utils"] is sentinel
    assert sys.path == original_path
    assert all(
        not (
            getattr(module, "__file__", None)
            and Path(module.__file__).resolve().is_relative_to(tmp_path)
        )
        for module in sys.modules.values()
        if isinstance(module, ModuleType)
    )


def test_upstream_loader_preserves_root_logging_configuration(tmp_path):
    entry_file = tmp_path / "inference.py"
    entry_file.write_text(
        "import logging\nlogging.basicConfig(level=logging.DEBUG)\n",
        encoding="utf-8",
    )
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    for handler in original_handlers:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)

    try:
        _load_upstream_module(entry_file, tmp_path)

        assert root_logger.handlers == []
        assert root_logger.level == logging.WARNING
    finally:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)


def test_upstream_runtime_logger_is_isolated_from_root(tmp_path):
    utils_module = ModuleType("utils")
    utils_module.__file__ = str(tmp_path / "utils.py")
    utils_module.logger = logging

    _isolate_upstream_logging({"utils": utils_module})

    assert isinstance(utils_module.logger, logging.Logger)
    assert utils_module.logger.name == "RealtimeTTS.Inflect.upstream"
    assert utils_module.logger.handlers


def test_module_containment_keeps_hugging_face_symlink_path(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    blob = tmp_path / "blob.py"
    blob.write_text("VALUE = 1\n", encoding="utf-8")
    model_file = snapshot / "model_file.py"
    try:
        model_file.symlink_to(blob)
    except OSError:
        pytest.skip("This host does not permit test symlinks")
    module = ModuleType("model_file")
    module.__file__ = str(model_file)

    assert not model_file.resolve().is_relative_to(snapshot)
    assert _module_is_below(module, snapshot)


def test_engine_identity_and_stream_info(engine_factory, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyaudio", SimpleNamespace(paInt16=8))
    engine = engine_factory()

    assert engine.engine_name == "inflect"
    assert engine.can_consume_generators is False
    assert engine.preload_sentence_tokenizer is True
    assert engine.get_stream_info() == (8, 1, 24_000)


def test_engine_uses_pinned_default_source(engine_factory):
    engine = engine_factory()

    assert engine.model_id == ONNX_MODEL_ID
    assert engine.revision == ONNX_REVISION


def test_verification_rejects_tampered_executable_helper(monkeypatch, tmp_path):
    relative = "runtime/text/cleaners.py"
    helper = tmp_path / relative
    helper.parent.mkdir(parents=True)
    helper.write_text("tampered = True\n", encoding="utf-8")
    monkeypatch.setitem(_PINNED_HASHES, "onnx", {relative: "0" * 64})
    engine = object.__new__(InflectEngine)
    engine.backend = "onnx"
    engine.model_dir = tmp_path
    engine.verify_files = True

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        engine._validate_runtime_files()


def test_synthesize_queues_clipped_pcm16(engine_factory, fake_runtime):
    fake_runtime.waveform = np.asarray([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    engine = engine_factory()

    assert engine.synthesize("  Hello   world. ") is True

    expected = (
        np.clip(fake_runtime.waveform, -1.0, 1.0) * 32767.0
    ).astype(np.int16).tobytes()
    assert engine.queue.get_nowait() == expected
    assert engine.audio_duration == pytest.approx(fake_runtime.waveform.size / 24_000)
    assert fake_runtime.calls == [
        {"text": "Hello world.", "speed": 1.0, "variation": 0.667, "seed": 0}
    ]


def test_synthesize_forwards_voice_parameters(engine_factory, fake_runtime):
    engine = engine_factory(speed=1.25, variation=0.2, seed=42)

    assert engine.synthesize("Parameters") is True
    assert fake_runtime.calls[-1] == {
        "text": "Parameters",
        "speed": 1.25,
        "variation": 0.2,
        "seed": 42,
    }


def test_engine_instances_share_the_global_synthesis_lock(engine_factory):
    first = engine_factory()
    second = engine_factory()

    assert first._synthesis_lock is second._synthesis_lock


def test_pytorch_rng_state_is_restored(engine_factory, fake_runtime):
    torch = pytest.importorskip("torch")

    engine = engine_factory(backend="pytorch")
    fake_runtime.on_synthesize = lambda: torch.manual_seed(999)
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()

    assert engine.synthesize("Preserve the host RNG") is True

    assert torch.equal(torch.random.get_rng_state(), before)


def test_stop_during_generation_discards_waveform(engine_factory, fake_runtime):
    engine = engine_factory()
    fake_runtime.on_synthesize = engine.stop

    assert engine.synthesize("Stop this") is True
    assert engine.queue.empty()
    assert engine.audio_duration == 0


def test_empty_text_is_success_without_model_call(engine_factory, fake_runtime):
    engine = engine_factory()

    assert engine.synthesize("   \n ") is True
    assert fake_runtime.calls == []
    assert engine.queue.empty()


def test_warmup_calls_runtime_without_queueing(
    monkeypatch, tmp_path, fake_runtime
):
    pytest.importorskip("torch")
    monkeypatch.setattr(InflectEngine, "_validate_runtime_files", lambda self: None)
    monkeypatch.setattr(InflectEngine, "_load_model", lambda self: fake_runtime)

    engine = InflectEngine(
        backend="pytorch",
        device="cpu",
        model_dir=tmp_path,
        warmup=True,
        warmup_text="  Ready   now. ",
        verify_files=False,
    )

    assert fake_runtime.calls == [
        {"text": "Ready now.", "speed": 1.0, "variation": 0.667, "seed": 0}
    ]
    assert engine.queue.empty()
    assert engine.audio_duration == 0


def test_pytorch_rng_state_is_restored_after_model_load(
    monkeypatch, tmp_path, fake_runtime
):
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(InflectEngine, "_validate_runtime_files", lambda self: None)

    def load_model(_engine):
        torch.manual_seed(999)
        return fake_runtime

    monkeypatch.setattr(InflectEngine, "_load_model", load_model)
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()

    InflectEngine(
        backend="pytorch",
        device="cpu",
        model_dir=tmp_path,
        warmup=False,
        verify_files=False,
    )

    assert torch.equal(torch.random.get_rng_state(), before)


def test_fixed_voice_and_parameters(engine_factory):
    engine = engine_factory(voice="default")

    assert engine.get_voices() == [InflectVoice("male")]
    assert repr(engine.get_voices()[0]) == "male"
    engine.set_voice(InflectVoice())
    engine.set_voice("inflect-micro-v2")
    engine.set_voice_parameters(speed=2.0, variation=0.0, seed=7)
    assert (engine.speed, engine.variation, engine.seed) == (2.0, 0.0, 7)

    with pytest.raises(ValueError, match="one fixed voice"):
        engine.set_voice("another")
    with pytest.raises(ValueError, match="speed"):
        engine.set_voice_parameters(speed=2.1)
    with pytest.raises(ValueError, match="variation"):
        engine.set_voice_parameters(variation=-0.1)
    with pytest.raises(TypeError, match="integer"):
        engine.set_voice_parameters(seed=1.5)
    with pytest.raises(ValueError, match="seed"):
        engine.set_voice_parameters(seed=-1)
    with pytest.raises(ValueError, match="Unsupported"):
        engine.set_voice_parameters(pitch=1)


def test_auto_backend_prefers_cuda_pytorch(monkeypatch):
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert InflectEngine._resolve_backend_and_device("auto", "auto") == (
        "pytorch",
        "cuda:0",
    )


def test_auto_backend_ignores_transitive_torch_for_onnx_extra(monkeypatch):
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "RealtimeTTS.engines.inflect_engine.importlib.util.find_spec",
        lambda name: object() if name == "onnxruntime" else None,
    )
    assert InflectEngine._resolve_backend_and_device("auto", "auto") == (
        "onnx",
        "cpu",
    )


def test_auto_backend_prefers_onnx_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "RealtimeTTS.engines.inflect_engine.importlib.util.find_spec",
        lambda name: object() if name == "onnxruntime" else None,
    )
    assert InflectEngine._resolve_backend_and_device("auto", "auto") == (
        "onnx",
        "cpu",
    )


def test_auto_backend_uses_pytorch_cpu_when_onnx_is_not_installed(monkeypatch):
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "RealtimeTTS.engines.inflect_engine.importlib.util.find_spec",
        lambda name: None,
    )
    assert InflectEngine._resolve_backend_and_device("auto", "auto") == (
        "pytorch",
        "cpu",
    )


def test_onnx_runtime_uses_tuned_session_options(tmp_path):
    sessions = []

    class FakeSessionOptions:
        pass

    class FakeOrt:
        class ExecutionMode:
            ORT_SEQUENTIAL = "sequential"

        class GraphOptimizationLevel:
            ORT_ENABLE_ALL = "all"

        SessionOptions = FakeSessionOptions

        @staticmethod
        def InferenceSession(path, *, sess_options, providers):
            session = SimpleNamespace(
                path=path,
                options=sess_options,
                providers=providers,
            )
            sessions.append(session)
            return session

    class FakeInflectONNX:
        pass

    entry_module = SimpleNamespace(
        ort=FakeOrt,
        InflectONNX=FakeInflectONNX,
        available_provider=lambda name: "CPUExecutionProvider",
    )
    engine = object.__new__(InflectEngine)
    engine.device = "cpu"
    engine.cpu_threads = 8
    engine.model_dir = tmp_path

    runtime = engine._create_onnx_runtime(entry_module)

    assert runtime.duration is sessions[0]
    assert runtime.decode is sessions[1]
    assert [Path(session.path).name for session in sessions] == [
        "duration.onnx",
        "decode.onnx",
    ]
    assert sessions[0].options is sessions[1].options
    assert sessions[0].options.intra_op_num_threads == 8
    assert sessions[0].options.inter_op_num_threads == 1
    assert sessions[0].options.execution_mode == "sequential"
    assert sessions[0].options.graph_optimization_level == "all"
    assert sessions[0].providers == ["CPUExecutionProvider"]


def test_shutdown_releases_runtime(engine_factory):
    engine = engine_factory()

    engine.shutdown()

    assert engine._runtime is None
    assert engine.synthesize("After shutdown") is False
