"""OpenAIEngine must hand audio chunks to the queue as they arrive, not after the whole response."""

import sys
import time
from types import ModuleType

import pytest

pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")
if "pyaudio" not in sys.modules:
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        stub = ModuleType("pyaudio")
        stub.paCustomFormat = 0
        stub.paInt16 = 8
        sys.modules["pyaudio"] = stub

from openai import OpenAI  # noqa: E402

from RealtimeTTS.engines.openai_engine import OpenAIEngine  # noqa: E402

CHUNKS = 4
DELAY = 0.2


class _SlowAudio(httpx.SyncByteStream):
    def __iter__(self):
        for _ in range(CHUNKS):
            time.sleep(DELAY)
            yield b"\xff" * 512


class _TimedQueue:
    def __init__(self):
        self.times = []

    def put(self, data):
        self.times.append(time.monotonic())


def test_first_chunk_is_queued_before_the_response_finishes():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"content-type": "audio/mpeg"}, stream=_SlowAudio()
        )
    )
    engine = OpenAIEngine(api_key="test")
    engine.client = OpenAI(
        api_key="test",
        base_url="http://test/v1",
        http_client=httpx.Client(transport=transport),
    )
    engine.queue = _TimedQueue()

    start = time.monotonic()
    assert engine.synthesize("Hello there.")

    assert len(engine.queue.times) == CHUNKS
    # The first chunk is sent after DELAY; buffering the body would delay it to CHUNKS * DELAY.
    assert engine.queue.times[0] - start < (CHUNKS - 1) * DELAY
