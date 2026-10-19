from types import SimpleNamespace
from RealtimeTTS.qwen_server import _alignment_fragment_metadata


def test_breeze_sound_tags_are_removed_from_alignment_not_source_text():
    engine = SimpleNamespace(alignment_non_speech_ranges=lambda text: [{"start": 6, "end": 15}])
    source = "Hello (chuckle), friend."
    meta = _alignment_fragment_metadata(engine, source, source, None)
    assert meta["text"] == "Hello , friend."
    assert meta["source_text"] == source
    assert meta["source_ranges"][8] == [17, 18]
    assert len(meta["source_ranges"]) == len(meta["text"])


def test_breeze_metadata_composes_existing_text_exchange_ranges():
    engine = SimpleNamespace(alignment_non_speech_ranges=lambda text: [{"start": 3, "end": 10}])
    source = "Hi (laugh) 12."
    text = "Hi (laugh) twelve."
    ranges = [[i, i + 1] for i in range(11)] + [[11, 13]] * 6 + [[13, 14]]
    meta = _alignment_fragment_metadata(engine, text, source, ranges)
    assert meta["text"] == "Hi  twelve."
    assert meta["source_text"] == source
    assert meta["source_ranges"][4:10] == [[11, 13]] * 6


def test_other_engines_keep_alignment_metadata_unchanged():
    assert _alignment_fragment_metadata(object(), "Hello.", "Hello.", None) == {"text": "Hello."}


def test_breeze_directions_are_not_highlighted_as_spoken_words():
    from RealtimeTTS.engines.breeze_engine import BreezeEngine
    source = "[Speak naturally] Hello!\n[Chuckle briefly] (chuckle)\n[Speak naturally] Welcome."
    meta = _alignment_fragment_metadata(BreezeEngine, source, source, None)
    assert meta["text"] == "Hello!\n\nWelcome."
    assert meta["source_text"] == source
    for letter, (start, end) in zip(meta["text"], meta["source_ranges"]):
        assert source[start:end] == letter
