import json
import inspect
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from RealtimeTTS import QwenEngine, QwenEngineError, QwenVoice
from RealtimeTTS.engines import qwen_engine as qwen_module


class FakeVoiceRef:
    def __init__(self, spk=None, codes=None):
        self.ref_spk_emb = np.asarray(
            [0.1, 0.2, 0.3] if spk is None else spk, dtype=np.float32
        )
        self.ref_codes = np.asarray(
            [[1, 2, 3], [4, 5, 6]] if codes is None else codes, dtype=np.int32
        )

    def save(self, spk_path, rvq_path):
        self.ref_spk_emb.tofile(spk_path)
        self.ref_codes.tofile(rvq_path)
        return Path(spk_path), Path(rvq_path)


class FakeLibrary:
    def version(self):
        return "fake-native-abi4"


class FakeBackend:
    def __init__(self, *, endless=False, delay=0.0, empty=False):
        self.library = FakeLibrary()
        self.endless = endless
        self.delay = delay
        self.empty = empty
        self.closed = False
        self.extract_calls = 0
        self.load_calls = 0
        self.stream_calls = []
        self.last_stream_profile = None
        self.stream_entered = threading.Event()
        self.cancel_observed = threading.Event()
        self.active_streams = 0
        self.max_active_streams = 0
        self._active_lock = threading.Lock()

    def extract_voice_ref(self, audio):
        self.extract_calls += 1
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        return FakeVoiceRef()

    def load_voice_ref(self, spk_path, rvq_path):
        self.load_calls += 1
        assert Path(spk_path).is_file()
        assert Path(rvq_path).is_file()
        return FakeVoiceRef()

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        self.stream_entered.set()
        cancel_event = kwargs["cancel_event"]
        callback_ns = time.perf_counter_ns()
        self.last_stream_profile = {
            "first_callback_perf_counter_ns": callback_ns,
            "first_callback_enter_ms": 0.0,
        }
        with self._active_lock:
            self.active_streams += 1
            self.max_active_streams = max(self.max_active_streams, self.active_streams)
        try:
            if self.endless:
                while not cancel_event.is_set():
                    time.sleep(0.002)
                    yield np.array([0.1, -0.1], dtype=np.float32), 24000
                if cancel_event.is_set():
                    self.cancel_observed.set()
                return
            if self.delay:
                time.sleep(self.delay)
            if self.empty:
                return
            yield np.array([0.0, 0.5, -0.5, np.nan, np.inf], dtype=np.float32), 24000
        finally:
            if cancel_event.is_set() and self.endless:
                self.cancel_observed.set()
            with self._active_lock:
                self.active_streams -= 1

    def close(self):
        self.closed = True


class SequenceBackend(FakeBackend):
    """Small deterministic native-stream stand-in for startup-buffer tests."""

    def __init__(self, chunks):
        super().__init__()
        self.chunks = [np.asarray(chunk, dtype=np.float32) for chunk in chunks]

    def stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        self.stream_entered.set()
        cancel_event = kwargs["cancel_event"]
        self.last_stream_profile = {"first_callback_enter_ms": 0.0}
        for chunk in self.chunks:
            if cancel_event.is_set():
                return
            yield chunk, 24000


def _reference_file(tmp_path, name="reference.wav"):
    path = tmp_path / name
    path.write_bytes(b"deterministic fake wav bytes")
    return path


def _engine(tmp_path, backend, voice=None, **kwargs):
    factory = kwargs.pop("backend_factory", lambda **_factory_kwargs: backend)
    return QwenEngine(
        voice=voice,
        voice_cache_dir=tmp_path / "voice-cache",
        warmup=False,
        backend_factory=factory,
        **kwargs,
    )


def _queued_pcm(engine):
    chunks = []
    while not engine.queue.empty():
        chunks.append(np.frombuffer(engine.queue.get_nowait(), dtype="<i2").copy())
    return np.concatenate(chunks) if chunks else np.empty(0, dtype="<i2")


def test_voice_selects_icl_only_when_reference_text_exists(tmp_path):
    wav = _reference_file(tmp_path)
    assert QwenVoice("xvec", ref_audio=wav).clone_mode == "x_vector"
    assert QwenVoice("icl", ref_audio=wav, ref_text=" exact words ").clone_mode == "icl"
    with pytest.raises(ValueError, match="spk_path and rvq_path"):
        QwenVoice("bad", spk_path=tmp_path / "only.spk")


def test_preencoded_voice_refs_are_cached_in_memory_for_fast_switching(tmp_path):
    spk = tmp_path / "voice.spk"
    rvq = tmp_path / "voice.rvq"
    spk.write_bytes(b"speaker")
    rvq.write_bytes(b"codes")
    backend = FakeBackend()
    engine = _engine(tmp_path, backend)
    engine.set_voice(QwenVoice("first", spk_path=spk, rvq_path=rvq))
    engine.set_voice(QwenVoice("second", spk_path=spk, rvq_path=rvq))
    assert backend.load_calls == 1
    engine.shutdown()


def test_pcm_conversion_clips_and_sanitizes_non_finite_values():
    pcm = qwen_module._float_to_pcm16(
        np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, np.nan], dtype=np.float32)
    )
    values = np.frombuffer(pcm, dtype="<i2")
    assert values.tolist() == [-32767, -32767, -16384, 0, 16384, 32767, 32767, 0]


def test_generated_voice_cache_is_atomic_and_content_versioned(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen_module,
        "_load_reference_audio",
        lambda _path: np.array([0.0, 0.1, -0.1], dtype=np.float32),
    )
    wav = _reference_file(tmp_path)
    first_backend = FakeBackend()
    first = _engine(tmp_path, first_backend)
    first.set_voice(QwenVoice("mira", ref_audio=wav, ref_text="hello"))
    assert first_backend.extract_calls == 1
    metadata_files = list((tmp_path / "voice-cache").rglob("*.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["audio_sha256"]
    assert metadata["ref_text"] == "hello"
    assert metadata["model_id"] == qwen_module.DEFAULT_MODEL
    assert metadata["quant"] == "Q8_0"
    assert metadata["native_abi"] == 4
    assert not list((tmp_path / "voice-cache").rglob("*.tmp"))
    assert not list((tmp_path / "voice-cache").rglob("*.lock"))
    first.shutdown()

    second_backend = FakeBackend()
    second = _engine(tmp_path, second_backend)
    second.set_voice(QwenVoice("mira", ref_audio=wav, ref_text="hello"))
    assert second_backend.extract_calls == 0
    assert second_backend.load_calls == 1
    second.shutdown()

    next((tmp_path / "voice-cache").rglob("*.spk")).write_bytes(b"corrupt")
    third_backend = FakeBackend()
    third = _engine(tmp_path, third_backend)
    third.set_voice(QwenVoice("mira repaired", ref_audio=wav, ref_text="hello"))
    assert third_backend.extract_calls == 1

    third.set_voice(QwenVoice("mira changed", ref_audio=wav, ref_text="different"))
    assert third_backend.extract_calls == 2
    assert len(list((tmp_path / "voice-cache").rglob("*.json"))) == 2
    third.shutdown()


def test_voice_cache_lock_serializes_competing_writers(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen_module,
        "_load_reference_audio",
        lambda _path: np.array([0.0, 0.1], dtype=np.float32),
    )
    wav = _reference_file(tmp_path)
    backends = [FakeBackend(delay=0.02), FakeBackend(delay=0.02)]
    engines = [_engine(tmp_path, backend) for backend in backends]
    errors = []

    def set_voice(engine):
        try:
            engine.set_voice(QwenVoice("shared", ref_audio=wav, ref_text="same"))
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=set_voice, args=(engine,)) for engine in engines]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
    assert not any(worker.is_alive() for worker in workers)
    assert not errors
    assert sum(backend.extract_calls for backend in backends) == 1
    assert sum(backend.load_calls for backend in backends) == 1
    assert len(list((tmp_path / "voice-cache").rglob("*.json"))) == 1
    assert not list((tmp_path / "voice-cache").rglob("*.lock"))
    for engine in engines:
        engine.shutdown()


@pytest.mark.parametrize("ref_text, expect_icl", [(None, False), ("exact transcript", True)])
def test_streams_24khz_pcm_and_selects_clone_mode(tmp_path, ref_text, expect_icl, monkeypatch):
    monkeypatch.setattr(
        qwen_module,
        "_load_reference_audio",
        lambda _path: np.array([0.0, 0.1], dtype=np.float32),
    )
    backend = FakeBackend()
    voice = QwenVoice(
        "voice",
        ref_audio=_reference_file(tmp_path),
        ref_text=ref_text,
        language="Auto",
    )
    engine = _engine(
        tmp_path,
        backend,
        voice=voice,
        seed=42,
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        repetition_penalty=1.1,
        trim_silence=False,
    )
    try:
        # This exercises the real BaseInitMeta-created queue/event, not constructor state.
        assert engine.synthesize("hello") is True
        assert engine.get_stream_info()[1:] == (1, 24000)
        pcm = np.frombuffer(engine.queue.get_nowait(), dtype="<i2")
        assert pcm.tolist() == [0, 16384, -16384, 0, 32767]
        call = backend.stream_calls[-1]
        assert call["lang"] == "auto"
        assert (call["ref_codes"] is not None) is expect_icl
        assert call["ref_text"] == (ref_text if expect_icl else None)
        assert call["seed"] == 42
        assert call["temperature"] == 0.7
        assert call["top_k"] == 20
        assert call["top_p"] == 0.8
        assert call["repetition_penalty"] == 1.1
        assert engine.audio_duration == pytest.approx(5 / 24000)
        assert engine.last_synthesis_profile["callback_to_queue_ms"] >= 0
        assert engine.last_synthesis_profile["first_chunk_duration_ms"] == pytest.approx(5 / 24)
        assert engine.last_synthesis_profile["predicted_underruns"] == 0
    finally:
        engine.shutdown()


def test_streaming_start_trim_uses_subframe_preroll_and_fade_in(tmp_path, monkeypatch):
    """Leading windows are trimmed without cutting directly into speech."""
    monkeypatch.setattr(
        qwen_module,
        "_load_reference_audio",
        lambda _path: np.array([0.0, 0.1], dtype=np.float32),
    )

    # The 0.004 section is below the 0.005 threshold. Speech begins in the
    # sixth 5 ms window of the second native callback. This catches accidental
    # whole-frame trimming and verifies that the detector works across chunks.
    chunks = [
        np.zeros(1920, dtype=np.float32),
        np.concatenate(
                [
                    np.zeros(120, dtype=np.float32),
                    np.full(120, 0.004, dtype=np.float32),
                    np.full(120, 0.004, dtype=np.float32),
                    np.full(120, 0.004, dtype=np.float32),
                    np.full(120, 0.004, dtype=np.float32),
                    np.full(120, 0.4, dtype=np.float32),
                    np.zeros(1920 - 720, dtype=np.float32),
            ]
        ),
    ]
    chunks.extend(np.full(1920, 0.4, dtype=np.float32) for _ in range(20))
    chunks.append(np.zeros(1920, dtype=np.float32))
    backend = SequenceBackend(chunks)

    engine = _engine(
        tmp_path,
        backend,
        voice=QwenVoice("voice", ref_audio=_reference_file(tmp_path)),
    )
    try:
        assert engine.synthesize("hello") is True
        queued = []
        while not engine.queue.empty():
            queued.append(np.frombuffer(engine.queue.get_nowait(), dtype="<i2").copy())
        audio = np.concatenate(queued)
        profile = engine.last_synthesis_profile

        # The trim is sub-frame precise: it removes some, but not all, of the
        # 2,520 samples before the first above-threshold window. It must not
        # degenerate into dropping one complete native callback.
        assert 0 < profile["leading_trimmed_samples"] < 2520
        assert profile["leading_trimmed_samples"] != 1920
        assert profile["leading_trimmed_ms"] == pytest.approx(
            profile["leading_trimmed_samples"] * 1000 / 24000
        )

        # A fade-in starts at zero and prevents a hard first-sample jump. The
        # first speech-containing window still survives the trim.
        assert audio[0] == 0
        assert np.max(np.abs(np.diff(audio.astype(np.int32)))) < 16000
        assert np.max(np.abs(audio)) > 10000
        assert profile["startup_fade_samples"] == 480
        assert profile["n_samples"] == audio.size
    finally:
        engine.shutdown()


def test_streaming_start_accumulates_a_safe_first_chunk_then_streams(tmp_path, monkeypatch):
    """The first published chunk is large enough for immediate playback."""
    monkeypatch.setattr(
        qwen_module,
        "_load_reference_audio",
        lambda _path: np.array([0.0, 0.1], dtype=np.float32),
    )
    backend = SequenceBackend([np.full(240, 0.25, dtype=np.float32) for _ in range(20)])
    engine = _engine(
        tmp_path,
        backend,
        voice=QwenVoice("voice", ref_audio=_reference_file(tmp_path)),
    )
    try:
        assert engine.synthesize("hello") is True
        queued = []
        while not engine.queue.empty():
            queued.append(np.frombuffer(engine.queue.get_nowait(), dtype="<i2").copy())
        assert len(queued) >= 2
        first_ms = len(queued[0]) * 1000 / 24000
        assert 150 <= first_ms <= 240
        assert engine.last_synthesis_profile["first_chunk_duration_ms"] == pytest.approx(
            first_ms
        )
        assert sum(len(chunk) for chunk in queued) == 20 * 240
    finally:
        engine.shutdown()


def test_streaming_start_flushes_a_short_output_at_end(tmp_path, monkeypatch):
    """A short utterance is not lost merely because it misses the startup target."""
    monkeypatch.setattr(
        qwen_module,
        "_load_reference_audio",
        lambda _path: np.array([0.0, 0.1], dtype=np.float32),
    )
    backend = SequenceBackend(
        [np.zeros(240, dtype=np.float32), np.full(120, 0.25, dtype=np.float32)]
    )
    engine = _engine(
        tmp_path,
        backend,
        voice=QwenVoice("voice", ref_audio=_reference_file(tmp_path)),
    )
    try:
        assert engine.synthesize("short") is True
        audio = _queued_pcm(engine)
        assert audio.size > 0
        assert np.max(np.abs(audio)) > 1000
        assert engine.last_synthesis_profile["n_samples"] == audio.size
    finally:
        engine.shutdown()


def test_streaming_start_trim_can_be_disabled_without_mutating_pcm(tmp_path, monkeypatch):
    """Disabling trimming keeps leading silence and does not apply a fade."""
    monkeypatch.setattr(
        qwen_module,
        "_load_reference_audio",
        lambda _path: np.array([0.0, 0.1], dtype=np.float32),
    )
    native = np.concatenate(
        [
            np.zeros(240, dtype=np.float32),
            np.full(240, 0.25, dtype=np.float32),
            np.zeros(120, dtype=np.float32),
        ]
    )
    backend = SequenceBackend([native])
    engine = _engine(
        tmp_path,
        backend,
        voice=QwenVoice("voice", ref_audio=_reference_file(tmp_path)),
        trim_silence=False,
    )
    try:
        assert engine.synthesize("hello") is True
        audio = _queued_pcm(engine)
        expected = qwen_module._float_to_pcm16(native)
        assert audio.tobytes() == expected
        assert engine.last_synthesis_profile["leading_trimmed_samples"] == 0
    finally:
        engine.shutdown()


def test_warmup_consumes_native_audio_without_publishing_it(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen_module,
        "_load_reference_audio",
        lambda _path: np.array([0.0, 0.1], dtype=np.float32),
    )
    backend = FakeBackend()
    engine = QwenEngine(
        voice=QwenVoice("voice", ref_audio=_reference_file(tmp_path)),
        voice_cache_dir=tmp_path / "cache",
        warmup=True,
        backend_factory=lambda **_kwargs: backend,
    )
    try:
        assert len(backend.stream_calls) == 1
        assert backend.stream_calls[0]["max_new_tokens"] == 16
        assert engine.queue.empty()
        assert engine.engine_name == "qwen"
    finally:
        engine.shutdown()


def test_stop_signals_native_cancel_and_context_is_reusable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen_module,
        "_load_reference_audio",
        lambda _path: np.array([0.0, 0.1], dtype=np.float32),
    )
    backend = FakeBackend(endless=True)
    engine = _engine(
        tmp_path,
        backend,
        voice=QwenVoice("voice", ref_audio=_reference_file(tmp_path)),
    )
    result = []
    worker = threading.Thread(target=lambda: result.append(engine.synthesize("long text")))
    worker.start()
    assert backend.stream_entered.wait(timeout=1)
    engine.stop()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert backend.cancel_observed.wait(timeout=1)
    assert result == [True]

    backend.endless = False
    assert engine.synthesize("reused") is True
    assert not engine.queue.empty()
    engine.shutdown()
    assert backend.closed is True
    assert engine.synthesize("after close") is False


def test_concurrent_syntheses_are_serialized_per_context(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen_module,
        "_load_reference_audio",
        lambda _path: np.array([0.0, 0.1], dtype=np.float32),
    )
    backend = FakeBackend(delay=0.03)
    engine = _engine(
        tmp_path,
        backend,
        voice=QwenVoice("voice", ref_audio=_reference_file(tmp_path)),
    )
    results = []
    workers = [
        threading.Thread(target=lambda text=text: results.append(engine.synthesize(text)))
        for text in ("first", "second")
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2)
    assert results == [True, True]
    assert backend.max_active_streams == 1
    engine.shutdown()


def test_preencoded_voice_skips_extraction(tmp_path):
    spk = tmp_path / "voice.spk"
    rvq = tmp_path / "voice.rvq"
    spk.write_bytes(b"spk")
    rvq.write_bytes(b"rvq")
    backend = FakeBackend()
    engine = _engine(tmp_path, backend, voice=QwenVoice("cached", spk_path=spk, rvq_path=rvq))
    try:
        assert backend.load_calls == 1
        assert backend.extract_calls == 0
    finally:
        engine.shutdown()


def test_empty_native_result_is_a_clear_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen_module,
        "_load_reference_audio",
        lambda _path: np.array([0.0, 0.1], dtype=np.float32),
    )
    engine = _engine(
        tmp_path,
        FakeBackend(empty=True),
        voice=QwenVoice("voice", ref_audio=_reference_file(tmp_path)),
    )
    try:
        assert engine.synthesize("hello") is False
        assert "without producing audio" in str(engine.last_error)
    finally:
        engine.shutdown()


def test_explicit_gguf_paths_are_forwarded_and_content_keyed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen_module,
        "_load_reference_audio",
        lambda _path: np.array([0.0, 0.1], dtype=np.float32),
    )
    talker = tmp_path / "talker.gguf"
    codec = tmp_path / "codec.gguf"
    talker.write_bytes(b"talker weights")
    codec.write_bytes(b"codec weights")
    captured = {}
    backend = FakeBackend()

    def factory(**kwargs):
        captured.update(kwargs)
        return backend

    engine = QwenEngine(
        talker_path=talker,
        codec_path=codec,
        voice_cache_dir=tmp_path / "cache",
        warmup=False,
        backend_factory=factory,
    )
    try:
        assert captured["talker_path"] == talker.resolve()
        assert captured["codec_path"] == codec.resolve()
        assert captured["clamp_fp16"] is True
        engine.set_voice(QwenVoice("voice", ref_audio=_reference_file(tmp_path)))
        metadata_path = next((tmp_path / "cache").rglob("*.json"))
        source = json.loads(metadata_path.read_text(encoding="utf-8"))["model_source"]
        assert source["kind"] == "explicit_gguf"
        assert source["talker_sha256"] == qwen_module._hash_file(talker)
        assert source["codec_sha256"] == qwen_module._hash_file(codec)
    finally:
        engine.shutdown()

    with pytest.raises(ValueError, match="supplied together"):
        QwenEngine(
            talker_path=talker,
            warmup=False,
            backend_factory=lambda **_kwargs: FakeBackend(),
        )


def test_native_loader_error_is_actionable(tmp_path):
    with pytest.raises(QwenEngineError, match="doctor"):
        _engine(
            tmp_path,
            backend=None,
            backend_factory=lambda **_kwargs: (_ for _ in ()).throw(OSError("qwen.dll missing")),
        )


@pytest.mark.parametrize(
    "native_message, expected",
    [
        ("CUDA out of memory", "VRAM"),
        ("native ABI version mismatch", "ABI mismatch"),
        ("CUDA driver version is insufficient", "CUDA initialization"),
    ],
)
def test_native_failures_are_translated(tmp_path, native_message, expected):
    with pytest.raises(QwenEngineError, match=expected):
        _engine(
            tmp_path,
            backend=None,
            backend_factory=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError(native_message)
            ),
        )


def test_public_exports_and_install_extra_are_declared():
    import RealtimeTTS
    import RealtimeTTS.engines

    assert RealtimeTTS.QwenVoice is QwenVoice
    assert RealtimeTTS.engines.QwenEngine is QwenEngine
    version_text = (Path(__file__).parents[1] / "RealtimeTTS" / "_version.py").read_text(
        encoding="utf-8"
    )
    assert f'__version__ = "{RealtimeTTS.__version__}"' in version_text
    setup_text = (Path(__file__).parents[1] / "setup.py").read_text(encoding="utf-8")
    assert '"qwen": base_requirements + qwen_requirements' in setup_text
    assert '"qwen-server": base_requirements + qwen_common_requirements' in setup_text
    assert (
        'realtimetts-qwen-native[cuda12]==0.1.0; sys_platform == "win32"'
        in setup_text
    )
    assert (
        'realtimetts-qwen-native[cuda12]==0.1.0; sys_platform == "linux"'
        in setup_text
    )
    requirements_text = (Path(__file__).parents[1] / "requirements.txt").read_text(
        encoding="utf-8"
    )
    assert (
        'realtimetts-qwen-native[cuda12]==0.1.0; sys_platform == "win32"'
        in requirements_text
    )
    assert (
        'realtimetts-qwen-native[cuda12]==0.1.0; sys_platform == "linux"'
        in requirements_text
    )
    assert "uvicorn[standard]>=0.34,<1" in requirements_text


def test_installed_native_binding_has_required_abi4_contract():
    qwentts_cpp = pytest.importorskip("qwentts_cpp")
    assert qwentts_cpp.QT_ABI_VERSION == 4
    init_parameters = inspect.signature(qwentts_cpp.QwenTTS.from_pretrained).parameters
    assert {"max_batch", "codec_chunk_sec"} <= set(init_parameters)
    stream_parameters = inspect.signature(qwentts_cpp.QwenTTS.stream).parameters
    assert "cancel_event" in stream_parameters
    assert "codec_chunk_sec" not in stream_parameters
