from types import SimpleNamespace
import math
import queue
import struct
import threading
import time

import pytest

from RealtimeTTS.alignment.server import parse_alignment_request, scale_alignment_event
from RealtimeTTS.audio_tempo import TempoProcessor
from RealtimeTTS.qwen_server import (
    ApiError,
    RequestMetrics,
    SynthesisState,
    _CaptureQueue,
)
from tests.test_qwen_server import _persist_voice, _server


def test_speed_is_validated_and_carried_to_http_request(tmp_path):
    _persist_voice(tmp_path)
    server = _server(tmp_path)
    request = server.parse_speech(
        {"input": "hello", "voice": "mira", "speed": 1.5}
    )
    assert request.speed == 1.5
    for value in (0.24, 4.01, float("nan"), "fast", True):
        with pytest.raises(ApiError, match="speed"):
            server.parse_speech({"input": "hello", "voice": "mira", "speed": value})


def test_capture_queue_flushes_tempo_output_before_completion():
    class FakeTempo:
        def __init__(self):
            self.inputs = []
            self.finished = False
            self.cancelled = False

        def feed(self, payload):
            self.inputs.append(bytes(payload))
            yield b"processed:" + bytes(payload)

        def finish(self):
            self.finished = True
            yield b"tail"

        def cancel(self):
            self.cancelled = True

    state = SynthesisState(
        "pcm",
        request_id="speed-test",
        output_queue_chunks=8,
        deadline=time.monotonic() + 1,
    )
    tempo = FakeTempo()
    queue_sink = _CaptureQueue(
        state,
        RequestMetrics(),
        abort=lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected abort")),
        max_output_bytes=1024,
        tempo_processor=tempo,
    )
    queue_sink.put(b"raw")
    queue_sink.finish()
    assert tempo.inputs == [b"raw"]
    assert tempo.finished
    assert state.output.get_nowait() == b"processed:raw"
    assert state.output.get_nowait() == b"tail"
    assert state.received_bytes == len(b"processed:raw") + len(b"tail")


def test_server_tempo_changes_duration_and_1x_preserves_pcm():
    pcm = b''.join(struct.pack('<h', int(12000 * math.sin(2 * math.pi * 440 * i / 24000)))
                   for i in range(24000))

    def queue_for(state, tempo_processor=None):
        return _CaptureQueue(
            state,
            RequestMetrics(),
            abort=lambda *_args: (_ for _ in ()).throw(AssertionError('unexpected abort')),
            max_output_bytes=128 * 1024,
            tempo_processor=tempo_processor,
        )

    direct_state = SynthesisState('pcm', request_id='speed-1x', output_queue_chunks=8,
                                  deadline=time.monotonic() + 1)
    direct = queue_for(direct_state)
    direct.put(pcm)
    direct.finish()
    assert direct_state.output.get_nowait() == pcm

    processed_state = SynthesisState('pcm', request_id='speed-2x', output_queue_chunks=512,
                                     deadline=time.monotonic() + 1)
    processed = queue_for(processed_state, TempoProcessor(8, 1, 24000, 2.0))
    processed.put(pcm)
    processed.finish()
    changed = b''.join(processed_state.output.get_nowait() for _ in range(processed_state.output.qsize()))
    assert len(pcm) * 0.4 < len(changed) < len(pcm) * 0.65
    assert changed != pcm


def test_model_alignment_suppresses_native_engine_alignment_temporarily():
    from RealtimeTTS.alignment.server import ServerAlignment
    from RealtimeTTS.alignment.timeline import AlignmentConfig

    native = object()
    engine = SimpleNamespace(
        queue=queue.Queue(), timings=queue.Queue(), streaming_aligner=native,
        provides_word_timings=True, on_alignment=None,
        alignment_start_event=None, alignment_cancel_event=None,
        audio_duration=0.0, get_stream_info=lambda: (8, 1, 24000),
    )
    state = SimpleNamespace(
        alignment_done=threading.Event(), alignment_closed=threading.Event(),
        alignment_events=queue.Queue(), cancelled=threading.Event(),
        first_pcm_published=threading.Event(), response_format='pcm',
    )
    request = SimpleNamespace(
        alignment=AlignmentConfig(source='model', aligner=object()),
        language='english', text='hello', source_text=None, source_ranges=None, speed=1.0,
    )
    session = ServerAlignment(engine, request, state)
    try:
        assert engine.streaming_aligner is None
    finally:
        session.finish(success=False)
        session.controller.close()
    assert engine.streaming_aligner is native


def test_model_alignment_preserves_guard_backend_and_forwards_cutoff():
    from RealtimeTTS.alignment.server import ServerAlignment
    from RealtimeTTS.alignment.timeline import AlignmentConfig

    native = object()
    forwarded = []
    engine = SimpleNamespace(
        queue=queue.Queue(), timings=queue.Queue(), streaming_aligner=native,
        provides_word_timings=True, hallucination_guard=True, acoustic_guard=False,
        on_alignment=forwarded.append, alignment_start_event=None,
        alignment_cancel_event=None, audio_duration=0.0,
        get_stream_info=lambda: (8, 1, 24000),
    )
    state = SimpleNamespace(
        alignment_done=threading.Event(), alignment_closed=threading.Event(),
        alignment_events=queue.Queue(), cancelled=threading.Event(),
        first_pcm_published=threading.Event(), response_format='pcm',
    )
    request = SimpleNamespace(
        alignment=AlignmentConfig(source='model', aligner=object()),
        language='english', text='hello', source_text=None, source_ranges=None, speed=1.0,
    )
    session = ServerAlignment(engine, request, state)
    event = {
        'type': 'alignment_complete', 'completion_verified': True,
        'stop_requested': True, 'trim_end_time': 0.25,
    }
    try:
        assert engine.streaming_aligner is native
        engine.on_alignment(event)
        assert forwarded == [event]
        assert state.alignment_events.get_nowait() == event
    finally:
        session.finish(success=False)
        session.controller.close()
    assert engine.streaming_aligner is native


def test_alignment_auto_prefers_display_model_and_scales_timing():
    studio = object()
    native = SimpleNamespace(timestamp_levels=("word",))
    engine = SimpleNamespace(
        provides_word_timings=True,
        streaming_aligner=native,
        studio_aligner=studio,
    )
    config = parse_alignment_request(
        {"source": "auto", "level": "character", "mode": "live"},
        engine,
        "en",
    )
    assert config.source == "model"
    assert config.aligner is studio
    native_config = parse_alignment_request(
        {"source": "native", "level": "word"},
        engine,
        "en",
    )
    assert native_config.aligner is native
    event = {
        "type": "alignment",
        "words": [{"word": "hello", "start_time": 1.0, "end_time": 2.0}],
        "text_start": 0,
        "duration": 2.0,
        "sample_rate_hz": 24000,
    }
    scaled = scale_alignment_event(event, 2.0)
    assert scaled["words"][0]["start_time"] == 0.5
    assert scaled["words"][0]["end_time"] == 1.0
    assert scaled["duration"] == 1.0
    assert scaled["text_start"] == 0
    assert event["words"][0]["start_time"] == 1.0
