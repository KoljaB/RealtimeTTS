import asyncio
from types import SimpleNamespace

import RealtimeTTS.qwen_server as server


def test_string_only_splitter_receives_text_and_preserves_voice_boundary(monkeypatch):
    seen = []
    async def strings_only(source, **options):
        parts = []
        async for part in source:
            assert isinstance(part, str)
            seen.append(part)
            parts.append(part)
        if parts:
            yield "".join(parts)
    monkeypatch.setattr(server, "_NATIVE_AVAILABLE_TEXT_RUNS", False)
    monkeypatch.setattr(server, "generate_sentences_async", strings_only)

    async def check():
        items = asyncio.Queue()
        boundary = server._VoiceBoundary("voice")
        for part in ["First ", "sentence.", boundary, "Second.", server._TEXT_END]:
            items.put_nowait(part)
        result = [part async for part in server._stream_sentence_fragments(items)]
        assert result == ["First sentence.", boundary, "Second."]
        assert seen == ["First sentence.", "Second."]
    asyncio.run(check())


def test_real_published_splitter_flushes_before_waiting_for_more_text(monkeypatch):
    monkeypatch.setattr(server, "_NATIVE_AVAILABLE_TEXT_RUNS", False)

    async def check():
        items = asyncio.Queue()
        items.put_nowait("Narration before a tool call.")
        items.put_nowait(server._TEXT_FLUSH)
        fragments = server._stream_sentence_fragments(
            items, tokenizer="rule-based", minimum_sentence_length=1,
            minimum_first_fragment_length=1,
        )
        try:
            first = await asyncio.wait_for(anext(fragments), timeout=1)
            assert first == "Narration before a tool call."
        finally:
            await fragments.aclose()
    asyncio.run(check())


def test_typed_splitter_results_are_normalized_to_strings(monkeypatch):
    async def typed_results(source, **options):
        async for part in source:
            yield SimpleNamespace(text=server._sentence_text(part))
    monkeypatch.setattr(server, "_NATIVE_AVAILABLE_TEXT_RUNS", False)
    monkeypatch.setattr(server, "generate_sentences_async", typed_results)

    async def check():
        items = asyncio.Queue()
        items.put_nowait("A complete sentence.")
        items.put_nowait(server._TEXT_END)
        assert [part async for part in server._stream_sentence_fragments(items)] == [
            "A complete sentence."
        ]
    asyncio.run(check())


def test_available_run_caps_keep_order_and_run_identity(monkeypatch):
    monkeypatch.setattr(server, "MAX_AVAILABLE_RUN_ITEMS", 2)

    async def check():
        items = asyncio.Queue()
        for part in ["a", "b", "c", "d", "e", server._TEXT_END]:
            items.put_nowait(part)
        runs = [part async for part in server._stream_token_source(items)]
        assert [run.text for run in runs] == ["ab", "cd", "e"]
        # Both native and compatibility records are built from the same run id.
        assert len(runs) == 3
        if not server._NATIVE_AVAILABLE_TEXT_RUNS:
            assert [run.run_id for run in runs] == [1, 1, 1]
    asyncio.run(check())
