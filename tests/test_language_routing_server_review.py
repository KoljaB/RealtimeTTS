"""Focused regression coverage for routed native alignment and Breeze controls."""

from __future__ import annotations

import json
import threading

from fastapi.testclient import TestClient

from RealtimeTTS import LanguageRoute, LanguageRoutingEngine
from RealtimeTTS.engines.base_engine import BaseEngine
from RealtimeTTS.engines.breeze_engine import BreezeEngine, BreezeVoice
from RealtimeTTS.language_routing_server import LanguageRoutingServer
from RealtimeTTS.qwen_server import create_app


class NativeAlignmentProvider(BaseEngine):
    """Small PCM provider whose native alignment callback is observable."""

    def post_init(self):
        self.engine_name = "native"
        self.direct_voice_control = True
        self.instruction_control = True
        self.provides_word_timings = True
        self.on_alignment = None
        self.last_alignment = None
        self.alignment_release = threading.Event()
        self.alignment_thread = None
        self.calls = []

    def get_stream_info(self):
        return 8, 1, 24000

    def get_voices(self):
        return ["default"]

    def set_voice(self, voice):
        self.voice = voice

    def set_language(self, language):
        self.language = language

    def set_instructions(self, instructions):
        self.instructions = instructions

    def synthesize(self, text, sentence_count=0):
        self.calls.append((text, self.language, self.voice, self.instructions))
        self.queue.put(b"\x20\x03" * 80)
        self.last_alignment = object()
        callback = self.on_alignment
        if callable(callback):
            def emit_alignment():
                self.alignment_release.wait(2.0)
                callback({
                    "type": "alignment",
                    "engine_audio_offset": 0.0,
                    "words": [{"word": "Hello", "start_time": 0.0, "end_time": 0.1}],
                })
                callback({"type": "alignment_done", "complete": True})

            self.alignment_thread = threading.Thread(
                target=emit_alignment, daemon=True
            )
            self.alignment_thread.start()
        return True


def test_routed_native_alignment_is_forwarded_by_automatic_websocket_mode(tmp_path):
    provider = NativeAlignmentProvider()
    router = LanguageRoutingEngine({
        "en": [LanguageRoute(provider, voice="default")],
    })
    server = LanguageRoutingServer(
        router, providers={"native": provider}, voice_dir=tmp_path
    )

    events = []
    pcm = bytearray()
    with TestClient(create_app(server)) as client:
        with client.websocket_connect("/v1/audio/speech-stream") as websocket:
            websocket.send_json({
                "type": "config",
                "voice": "default",
                "language": "en",
                "response_format": "pcm",
                "split_sentences": False,
            })
            websocket.send_json({
                "type": "text",
                "text": "Hello",
                "language": "en",
            })
            websocket.send_json({"type": "end"})
            while True:
                message = websocket.receive()
                if message.get("bytes") is not None:
                    pcm.extend(message["bytes"])
                    provider.alignment_release.set()
                    continue
                event = json.loads(message["text"])
                assert event["type"] != "error", event
                events.append(event)
                if event["type"] == "done":
                    break

    assert pcm == b"\x20\x03" * 80
    assert [event["type"] for event in events].count("fragment_start") == 1
    assert any(
        event["type"] == "alignment"
        and event["words"][0]["word"] == "Hello"
        for event in events
    )
    assert any(event["type"] == "alignment_done" for event in events)
    assert provider.calls == [("Hello", "en", "default", "")]
    provider.alignment_release.set()
    provider.alignment_thread.join(timeout=1.0)
    assert not provider.alignment_thread.is_alive()


def _offline_breeze() -> BreezeEngine:
    """Construct only the local Breeze state; no worker or network is used."""
    engine = BreezeEngine.__new__(BreezeEngine)
    BaseEngine.__init__(engine)
    engine._voice_id = ""
    engine._prepared = {}
    engine._control = threading.RLock()
    engine._generation_lock = threading.Lock()
    engine._stream_context = None
    engine._cancel = threading.Event()
    engine._pause_requested = threading.Event()
    engine._paused = threading.Event()
    engine._idle = threading.Event()
    engine._idle.set()
    engine.last_error = None
    engine.last_metrics = {}
    engine.last_diagnostics = {}
    engine._cancel_requested_at = None
    engine.streaming_aligner = None
    engine._owns_aligner = False
    engine.alignment_model = None
    engine.alignment_language = None
    engine.alignment_threads = 2
    engine.alignment_start_event = None
    engine.alignment_cancel_event = None
    engine.on_alignment = None
    engine._delivery_instructions = None
    return engine


def test_breeze_language_and_delivery_controls_preserve_voice_design_identity():
    engine = _offline_breeze()
    voice = BreezeVoice(
        "designed",
        instruct="A clear adult narrator",
        language="english",
        voice_design=True,
    )
    engine.set_voice(voice)
    engine.set_language("Chinese")
    engine.set_instructions("warm and deliberate")

    captured = []

    def capture(text, instruction, **kwargs):
        captured.append((text, instruction))
        return True

    engine._speak_block = capture
    assert engine.synthesize("Hello.") is True
    assert engine._voice.language == "chinese"
    assert engine._voice.voice_design is True
    assert engine._voice.instruct == voice.instruct
    assert captured == [
        ("Hello.", "A clear adult narrator Delivery: warm and deliberate")
    ]

    captured.clear()
    engine.set_language("English")
    assert engine.synthesize("Again.") is True
    assert engine._voice.language == "english"
    assert engine._voice.voice_design is True
    assert engine._voice.instruct == voice.instruct
    assert captured == [
        ("Again.", "A clear adult narrator Delivery: warm and deliberate")
    ]
