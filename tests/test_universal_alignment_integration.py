"""Cross-layer tests for the public alignment API and existing Studio servers."""
import json
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from RealtimeTTS import AlignmentConfig, TextToAudioStream
from RealtimeTTS._audio_backend import pyaudio
from RealtimeTTS.alignment.base import WordAlignment
from RealtimeTTS.alignment.controller import AlignmentController
from RealtimeTTS.alignment.server import parse_alignment_request
from RealtimeTTS.engines.base_engine import TimingInfo
from RealtimeTTS.qwen_server import create_app
from tests.test_alignment_controller import FakeEngine, _pcm
from tests.test_text_to_stream_alignment import _PCMEngine, _FakePyAudio, _single_sentence
from tests.test_qwen_server import FakeEngine as ServerEngine, _server, _persist_voice
import RealtimeTTS.stream_player as player_module


class Words:
    def __init__(self):
        self.calls = []

    def align_words(self, text, audio, sample_rate):
        self.calls.append((text, len(audio), sample_rate))
        words = text.split()
        duration = len(audio) / sample_rate
        return [WordAlignment(word.strip("."), i * duration / len(words),
                              (i + 1) * duration / len(words))
                for i, word in enumerate(words)]


def test_sdk_generator_input_and_completed_exports(monkeypatch, tmp_path):
    monkeypatch.setattr(player_module.pyaudio, "PyAudio", _FakePyAudio)

    class GeneratorEngine(_PCMEngine):
        def post_init(self):
            super().post_init()
            self.can_consume_generators = True

        def synthesize(self, text):
            observed = "".join(text)
            assert observed == "Hello café."
            self.queue.put(np.ones(1600, dtype=np.int16).tobytes())
            return True

    engine = GeneratorEngine()
    original_queue = engine.queue
    model = Words()
    stream = TextToAudioStream(engine, alignment=AlignmentConfig(
        mode="export", aligner=model), muted=True)
    try:
        stream.feed(iter(["Hello ", "café."])).play()
        snapshot = stream.alignment.snapshot()
        assert snapshot.final and snapshot.status == "ready"
        assert snapshot.text == "Hello café."
        assert [(w.text_start, w.text_end) for w in snapshot.words] == [(0, 5), (6, 10)]
        assert snapshot.duration == pytest.approx(0.1)
        snapshot.export(tmp_path / "speech.json")
        snapshot.export(tmp_path / "speech.srt")
        snapshot.export(tmp_path / "speech.vtt")
        assert json.loads((tmp_path / "speech.json").read_text(encoding="utf-8"))["words"]
        assert "WEBVTT" in (tmp_path / "speech.vtt").read_text(encoding="utf-8")
        assert "00:00:00,000" in (tmp_path / "speech.srt").read_text(encoding="utf-8")
    finally:
        stream.close()
    assert engine.queue is original_queue


def test_current_word_uses_absolute_playback_position():
    engine = FakeEngine(native=True)
    controller = AlignmentController("auto")
    try:
        binding = controller.bind(engine)
        binding.begin("one two")
        engine.queue.put(_pcm(1))
        engine.timings.put(TimingInfo(0, 0.4, "one"))
        engine.timings.put(TimingInfo(0.4, 0.9, "two"))
        binding.finish()
        controller.finish()
        controller.wait(1)
        controller.advance(0.1)
        controller.advance(0.2)
        controller.advance(0.3)
        assert controller.current_word.text == "one"
        controller.advance(0.5)
        assert controller.current_word.text == "two"
    finally:
        controller.close()


@pytest.mark.parametrize('verified', [True, False])
def test_embedded_aligner_finishes_after_queue_detach_with_paragraph_offsets(verified):
    engine = FakeEngine(native=True, clock=10)
    engine.streaming_aligner = SimpleNamespace(timestamp_levels=("word", "character"))
    controller = AlignmentController(AlignmentConfig(level="character"))
    try:
        binding = controller.bind(engine)
        binding.begin("a b")
        callback = engine.on_alignment
        engine.queue.put(_pcm(1))
        done = threading.Event()
        engine.last_alignment = SimpleNamespace(done=done)
        binding.finish()
        binding.detach()
        controller.finish()
        for word, offset in [("a", 10.0), ("b", 10.5)]:
            callback({"type": "alignment", "engine_audio_offset": offset,
                      "words": [{"word": word, "start_time": 0, "end_time": 0.3}],
                      "characters": [{"character": word, "start_time": 0, "end_time": 0.3}]})
            callback({"type": "alignment_done", "engine_audio_offset": offset,
                      "completion_verified": verified if word == "a" else True})
        assert not controller.snapshot().final
        done.set()
        snapshot = controller.wait(1)
        assert snapshot.status == ("ready" if verified else "partial")
        assert [w.start_time for w in snapshot.words] == [0, 0.5]
        assert [c.start_time for c in snapshot.characters] == [0, 0.5]
        assert [c.text_start for c in snapshot.characters] == [0, 2]
    finally:
        controller.close()


def test_non_spoken_control_tags_preserve_source_offsets():
    engine = FakeEngine()
    engine.alignment_non_speech_ranges = lambda text: [{"start": 3, "end": 10}]
    model = Words()
    controller = AlignmentController(AlignmentConfig(aligner=model))
    try:
        binding = controller.bind(engine)
        binding.begin("Hi (laugh) friend.")
        engine.queue.put(_pcm(1))
        binding.finish()
        controller.finish()
        snapshot = controller.wait(1)
        assert model.calls[0][0] == "Hi  friend."
        assert snapshot.text == "Hi (laugh) friend."
        assert [(w.text_start, w.text_end) for w in snapshot.words] == [(0, 2), (11, 17)]
    finally:
        controller.close()


def test_synced_wait_budget_is_paid_once_per_fragment(monkeypatch):
    engine = FakeEngine()
    controller = AlignmentController(AlignmentConfig(aligner=Words(), mode="synced", max_buffer_seconds=0))
    try:
        binding = controller.bind(engine)
        binding.begin("hello")
        engine.queue.put(_pcm(1))
        calls = []
        original = controller.timeline.set_status
        monkeypatch.setattr(controller.timeline, "set_status",
                            lambda status: (calls.append(status), original(status))[-1])
        controller.before_playback(0)
        controller.before_playback(0.1)
        controller.before_playback(0.2)
        assert calls == ["behind"]
    finally:
        controller.close()


def test_cancelled_audio_cannot_mutate_next_timeline():
    engine = FakeEngine()
    controller = AlignmentController(AlignmentConfig(aligner=Words()))
    try:
        binding = controller.bind(engine)
        binding.begin("old")
        controller.cancel()
        controller.reset()
        engine.queue.put(_pcm(1))
        assert controller.snapshot().duration == 0
        assert engine.raw_queue.empty()
    finally:
        controller.close()


@pytest.mark.parametrize("native", [True, False])
def test_missing_timestamps_are_reported_as_unavailable(native):
    engine = FakeEngine(native=native)
    model = SimpleNamespace(align_words=lambda *_: [])
    controller = AlignmentController(AlignmentConfig(aligner=model))
    try:
        binding = controller.bind(engine)
        binding.begin("hello")
        engine.queue.put(_pcm())
        binding.finish()
        controller.finish()
        snapshot = controller.wait(1)
        assert snapshot.status == "unavailable"
        assert snapshot.errors
    finally:
        controller.close()


def test_remote_config_cannot_request_model_paths_or_load_an_unconfigured_model():
    engine = FakeEngine()
    with pytest.raises(ValueError, match="Unsupported alignment fields"):
        parse_alignment_request({"model_dir": "/arbitrary"}, engine, "english")
    with pytest.raises(ValueError, match="Configure an alignment backend"):
        parse_alignment_request("auto", engine, "english")


class SharedServerEngine(ServerEngine):
    def __init__(self):
        super().__init__()
        self.timings = queue.Queue()
        self.audio_duration = 0
        self.provides_word_timings = False
        self.streaming_aligner = Words()
        self.on_alignment = None

    def get_stream_info(self):
        return pyaudio.paInt16, 1, 24000


def test_existing_server_websocket_publishes_shared_timeline(tmp_path):
    engine = SharedServerEngine()
    _persist_voice(tmp_path)
    server = _server(tmp_path, engine)
    with TestClient(create_app(server)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as websocket:
            websocket.send_json({"type": "config", "voice": "mira",
                                 "response_format": "pcm",
                                 "alignment": {"source": "model", "mode": "export"}})
            websocket.send_json({"type": "text", "text": "Hello café."})
            websocket.send_json({"type": "end"})
            events, audio = [], bytearray()
            while True:
                message = websocket.receive()
                if message.get("bytes") is not None:
                    audio.extend(message["bytes"])
                    continue
                event = json.loads(message["text"])
                events.append(event)
                if event["type"] in {"done", "error"}:
                    break
    assert audio
    final = [e for e in events if e["type"] == "alignment" and e.get("final")]
    assert final and final[-1]["status"] == "ready", events
    assert final[-1]["text_offset_unit"] == "unicode_codepoint"
    assert final[-1]["words"][-1]["text_start"] == 6
    assert any(e["type"] == "alignment_done" for e in events)



@pytest.mark.parametrize("display_alignment", [False, True])
def test_verified_engine_cutoff_reaches_websocket_with_fragment_identity(tmp_path, display_alignment):
    class GuardedEngine(SharedServerEngine):
        def __init__(self):
            super().__init__()
            self.provides_word_timings = True
            self.streaming_aligner = SimpleNamespace(timestamp_levels=("word",))

        def synthesize(self, text):
            self.queue.put(np.tile(np.array([1000, -1000], dtype='<i2'), 6000).tobytes())
            self.on_alignment({"type": "alignment", "words": [
                {"word": "Hello", "start_time": 0, "end_time": .1}
            ]})
            self.on_alignment({
                "type": "alignment_complete", "completion_verified": True,
                "stop_requested": True, "end_time": .1, "trim_end_time": .35,
            })
            self.on_alignment({"type": "alignment_done", "complete": True})
            return True

    engine = GuardedEngine()
    _persist_voice(tmp_path)
    server = _server(tmp_path, engine)
    options = {"type": "config", "voice": "mira", "response_format": "pcm"}
    if display_alignment:
        options["alignment"] = {"source": "native", "level": "word"}
    with TestClient(create_app(server)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as websocket:
            websocket.send_json(options)
            websocket.send_json({"type": "text", "text": "Hello."})
            websocket.send_json({"type": "end"})
            events, audio = [], bytearray()
            while True:
                message = websocket.receive()
                if message.get("bytes") is not None:
                    audio.extend(message["bytes"])
                    continue
                event = json.loads(message["text"])
                events.append(event)
                if event["type"] in {"done", "error"}:
                    break
    assert audio and not any(e["type"] == "error" for e in events), events
    start = next(e for e in events if e["type"] == "fragment_start")
    cutoffs = [e for e in events if e["type"] == "alignment_complete"]
    assert len(cutoffs) == 1, events
    cutoff = cutoffs[0]
    assert cutoff["completion_verified"] and cutoff["stop_requested"]
    assert cutoff["trim_end_time"] == pytest.approx(.35)
    for name in ("session_id", "request_id", "fragment_index"):
        assert cutoff[name] == start[name]


def test_existing_http_endpoint_rejects_metadata_request(tmp_path):
    _persist_voice(tmp_path)
    with TestClient(create_app(_server(tmp_path, SharedServerEngine()))) as client:
        response = client.post("/v1/audio/speech", json={
            "input": "Hello", "voice": "mira", "alignment": "auto"})
        assert response.status_code == 400
        assert "WebSocket" in response.text


def test_explicit_alignment_language_is_preserved():
    config = AlignmentConfig(language="german")
    assert AlignmentConfig.coerce(config, language="en").language == "german"
    assert AlignmentConfig.coerce({"language": "german"}, language="en").language == "german"


def test_mp3_sdk_wav_header_and_timeline_agree_after_device_resampling(monkeypatch, tmp_path):
    import wave
    from tests.test_alignment_decoder import _mp3_fixture

    monkeypatch.setattr(player_module.pyaudio, "PyAudio", _FakePyAudio)

    def resample(samples, source, target):
        count = round(len(samples) * target / source)
        return np.interp(np.arange(count) * source / target, np.arange(len(samples)), samples)

    monkeypatch.setattr(player_module.resampy, "resample", resample)
    payload = _mp3_fixture()

    class MP3Engine(_PCMEngine):
        def get_stream_info(self):
            return pyaudio.paCustomFormat, -1, -1

        def synthesize(self, text, sentence_count=0):
            for start in range(0, len(payload), 137):
                self.queue.put(payload[start:start + 137])
            return True

    model = Words()
    stream = TextToAudioStream(MP3Engine(), muted=False, alignment=AlignmentConfig(
        mode="export", aligner=model))
    path = tmp_path / "decoded.wav"
    try:
        stream.feed("Hello.").play(tokenize_sentences=_single_sentence, output_wavfile=str(path))
        snapshot = stream.alignment.snapshot()
        with wave.open(str(path)) as output:
            assert output.getframerate() == 16000
            assert output.getsampwidth() == 2
            duration = output.getnframes() / output.getframerate()
        assert duration == pytest.approx(snapshot.duration, abs=0.01)
        assert snapshot.status == "ready"
        assert model.calls[0][2] == 24000
    finally:
        stream.close()


def test_cpu_adapter_aligns_acoustic_character_boundaries_and_rejects_wrong_language():
    from RealtimeTTS.alignment.backends import CPUAlignmentModel
    from RealtimeTTS.alignment.ctc import log_softmax

    class Backend:
        language = "english"
        frame_seconds = 0.02
        blank_id = 0
        timestamp_levels = ("word", "character")

        def encode(self, text):
            return "ab", [1, 2]

        def emissions(self, audio, sample_rate):
            logits = np.full((6, 3), -8, np.float32)
            logits[np.arange(6), [0, 1, 1, 0, 2, 0]] = 8
            return log_softmax(logits)

    backend = Backend()
    result = CPUAlignmentModel(backend, "en").align("ab", np.zeros(1000), 24000, level="character")
    assert [c["character"] for c in result["characters"]] == ["a", "b"]
    assert result["characters"][0]["start_time"] == pytest.approx(0.02)
    assert result["characters"][1]["start_time"] == pytest.approx(0.08)
    assert result["words"][0]["word"] == "ab"
    with pytest.raises(ValueError, match="supports english"):
        CPUAlignmentModel(backend, "de").align("ab", np.zeros(1000), 24000)


def test_shared_server_does_not_treat_engine_prefix_completion_as_final():
    from RealtimeTTS.alignment.server import ServerAlignment

    engine = FakeEngine(native=True)
    engine.streaming_aligner = SimpleNamespace(timestamp_levels=("word", "character"))
    state = SimpleNamespace(
        alignment_done=threading.Event(), alignment_closed=threading.Event(),
        alignment_events=queue.Queue(), cancelled=threading.Event(),
        first_pcm_published=threading.Event(), response_format="pcm",
    )
    request = SimpleNamespace(
        alignment=AlignmentConfig(), language="english", text="hello",
        source_text=None, source_ranges=None,
    )
    session = ServerAlignment(engine, request, state)
    try:
        engine.on_alignment({"type": "alignment_complete", "complete": True})
        with state.alignment_events.mutex:
            assert not any(event["type"] == "alignment_complete"
                           for event in state.alignment_events.queue)
        assert not state.alignment_done.is_set()
        assert not session.controller.snapshot().final
    finally:
        session.finish(success=False)
        session.controller.close()
