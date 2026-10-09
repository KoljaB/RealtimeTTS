"""Request-scoped whole-text synthesis without changing realtime defaults."""
import asyncio
import json

import pytest

from RealtimeTTS import qwen_server as server
from tests.test_breeze_engine_server import server_client


TEXT = "This is the first complete sentence. This is the second complete sentence!"


def receive_done(ws):
    events, pcm = [], b""
    while True:
        message = ws.receive()
        assert message["type"] != "websocket.close", message
        if message.get("bytes"):
            pcm += message["bytes"]
        elif message.get("text"):
            event = json.loads(message["text"])
            assert event["type"] != "error", event
            events.append(event)
            if event["type"] == "done":
                return events, pcm


@pytest.mark.parametrize("split", [None, True, False])
def test_websocket_option_changes_actual_engine_calls(tmp_path, split):
    engine, client = server_client(tmp_path)
    # Fragment lifecycle events are emitted only when alignment is enabled.
    engine.streaming_aligner = object()
    with client, client.websocket_connect("/v1/audio/speech-stream") as ws:
        config = {"type": "config", "voice": "test"}
        if split is not None:
            config["split_sentences"] = split
        ws.send_json(config)
        for text in (TEXT[:18], TEXT[18:40], TEXT[40:]):
            ws.send_json({"type": "text", "text": text})
        ws.send_json({"type": "end"})
        events, pcm = receive_done(ws)
    assert pcm
    calls = [text for text, _instructions in engine.seen]
    if split is False:
        assert calls == [TEXT]
    else:
        assert len(calls) > 1
        assert " ".join(calls) == TEXT
    assert len([event for event in events if event["type"] == "fragment_start"]) == len(calls)


@pytest.mark.parametrize("format", ["pcm", "wav"])
def test_http_remains_whole_text(tmp_path, format):
    engine, client = server_client(tmp_path)
    with client:
        assert client.get("/v1/capabilities").json()["features"]["sentence_splitting"] is True
        response = client.post("/v1/audio/speech", json={
            "voice": "test", "input": TEXT, "split_sentences": False, "response_format": format,
        })
    assert response.status_code == 200 and response.content
    assert engine.seen == [(TEXT, None)]


@pytest.mark.parametrize("value", ["false", 0, None])
def test_nonboolean_option_rejected(tmp_path, value):
    _engine, client = server_client(tmp_path)
    with client:
        response = client.post("/v1/audio/speech", json={
            "voice": "test", "input": TEXT, "split_sentences": value,
        })
        assert response.status_code == 400
        assert "boolean" in response.json()["error"]["message"]
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            ws.send_json({"type": "config", "voice": "test", "split_sentences": value})
            assert ws.receive_json()["type"] == "error"


def test_unsplit_retains_explicit_boundaries_and_whitespace():
    async def collect():
        items = asyncio.Queue()
        boundaries = [
            server._TEXT_FLUSH, server._InstructionBoundary("Whisper."),
            server._VoiceBoundary("other"),
        ]
        expected = []
        for boundary in boundaries:
            text = "First sentence, with a comma.\nSecond sentence! "
            await items.put(text[:9])
            await items.put(text[9:])
            await items.put(boundary)
            expected.append(text)
            if boundary is not server._TEXT_FLUSH:
                expected.append(boundary)
        await items.put(server._TEXT_END)
        fragments = [part async for part in server._stream_sentence_fragments(
            items, split_sentences=False,
        )]
        assert fragments == expected
    asyncio.run(collect())


def test_unsplit_flush_synthesizes_before_end_and_instruction_change(tmp_path):
    engine, client = server_client(tmp_path)
    with client, client.websocket_connect("/v1/audio/speech-stream") as ws:
        ws.send_json({"type": "config", "voice": "test", "split_sentences": False})
        ws.send_json({"type": "segment_start", "segment_id": 1})
        ws.send_json({"type": "text", "text": TEXT, "instructions": "Whisper."})
        ws.send_json({"type": "flush", "segment_id": 1})
        while True:
            message = ws.receive()
            if message.get("text"):
                event = json.loads(message["text"])
                assert event["type"] != "error", event
                if event["type"] == "segment_done":
                    break
        assert engine.seen == [(TEXT, "Whisper.")]
        ws.send_json({"type": "text", "text": TEXT, "instructions": "Angry."})
        ws.send_json({"type": "end"})
        receive_done(ws)
    assert engine.seen == [(TEXT, "Whisper."), (TEXT, "Angry.")]
