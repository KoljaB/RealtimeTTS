"""Nonlinguistic splitter tails must not abort otherwise valid streamed speech."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from RealtimeTTS.qwen_server import _stream_sentence_fragments, _TEXT_END
from tests.test_qwen_server import FakeEngine, _server, _register, create_app
from tests.test_qwen_segmented_narration import config, segment, until

FILM = 'The film is open and playing — I can see "Coin Operated - Animated Short Film" by Two Ghosts, currently at 0:01 of 5:14. Enjoy the movie! 🎬'


def test_exact_live_sentence_has_isolated_emoji_tail():
    async def split():
        items = asyncio.Queue()
        await items.put(FILM)
        await items.put(_TEXT_END)
        return [fragment async for fragment in _stream_sentence_fragments(
            items, tokenizer="rule-based", language="en", minimum_sentence_length=10,
            minimum_first_fragment_length=10, quick_yield_single_sentence_fragment=True,
            quick_yield_for_all_sentences=True, quick_yield_every_fragment=False,
            force_first_fragment_after_words=1000000, fragment_lookahead_words=0)]
    assert asyncio.run(split())[-1] == "🎬"


@pytest.mark.parametrize("text", [FILM, "12345", "你好世界", "مرحبا", "नमस्ते", "🎬"])
def test_stream_skips_only_nonlinguistic_fragments_and_completes_segment(tmp_path, text):
    class Engine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.calls = []
        def synthesize(self, fragment):
            self.calls.append(fragment)
            assert any(character.isalnum() for character in fragment)
            return super().synthesize(fragment)
    engine = Engine()
    with TestClient(create_app(_server(tmp_path, engine))) as client:
        _register(client)
        with client.websocket_connect("/v1/audio/speech-stream") as ws:
            config(ws)
            segment(ws, 1, text)
            ws.send_json({"type": "end"})
            events = until(ws, "done")
            assert {"type": "segment_done", "segment_id": 1, "skipped": False} in events
    if text == "🎬":
        assert engine.calls == []
    else:
        assert engine.calls
        assert any(event["type"] == "pcm" for event in events)
        if text == FILM:
            assert engine.calls[-1] == "Enjoy the movie!"
