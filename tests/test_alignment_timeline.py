import json
import threading
import time

import pytest

from RealtimeTTS.alignment.timeline import (
    AlignedSpan,
    AlignmentConfig,
    AlignmentTimeline,
    project_text_offsets,
)


def _word(text, start, end, **kwargs):
    return AlignedSpan(text, start, end, level="word", **kwargs)


def _character(text, start, end, **kwargs):
    return AlignedSpan(text, start, end, level="character", **kwargs)


def test_config_coerce_and_validation():
    assert AlignmentConfig.coerce(None) is None
    assert AlignmentConfig.coerce(False) is None
    assert AlignmentConfig.coerce("off") is None
    config = AlignmentConfig.coerce(True, language="de", aligner="fake")
    assert config == AlignmentConfig(language="de", aligner="fake")
    assert AlignmentConfig().max_audio_seconds == 120.0
    assert AlignmentConfig.coerce({"mode": "export", "level": "character"}).mode == "export"
    with pytest.raises(ValueError):
        AlignmentConfig(source="decoder")
    with pytest.raises(ValueError):
        AlignmentConfig(mode="batch")
    with pytest.raises(ValueError):
        AlignmentConfig(level="phoneme")
    with pytest.raises(ValueError):
        AlignmentConfig(update_interval=0)
    with pytest.raises(ValueError):
        AlignmentConfig(max_pending=0)
    with pytest.raises(ValueError):
        AlignmentConfig(max_audio_seconds=0)
    with pytest.raises(ValueError):
        AlignmentConfig(max_audio_seconds=float("inf"))


def test_aligned_span_aliases_and_validation():
    span = _word("hello", 0, 0.5, score=0.9)
    assert span.word == "hello"
    assert span.character == "hello"
    assert span.to_dict()["word"] == "hello"
    with pytest.raises(ValueError):
        _word("bad", 0.4, 0.2)
    with pytest.raises(ValueError):
        _word("bad", float("nan"), 0.2)
    with pytest.raises(ValueError):
        _word("bad", 0, float("inf"))


def test_atomic_fragment_replacement_and_final_fence():
    updates = []
    timeline = AlignmentTimeline("hello world", on_update=updates.append)
    first = timeline.publish([_word("hello", 0, 0.2)], fragment_id="f1")
    timeline.publish([_word("other", 0.4, 0.6)], fragment_id="f2")
    second = timeline.publish([_word("world", 0.2, 0.4)], fragment_id="f1")
    assert [span.text for span in first.words] == ["hello"]
    assert [span.text for span in second.words] == ["world", "other"]
    assert [span.text for span in timeline.words] == ["world", "other"]
    assert updates[-1].words == (second.words[0], timeline.words[1])
    final = timeline.finish()
    revision = final.revision
    assert timeline.publish([_word("late", 0.4, 0.5)], fragment_id="f1").revision == revision
    assert [span.text for span in timeline.words] == ["world", "other"]


def test_reset_isolates_old_fragments_and_revision_is_monotonic():
    timeline = AlignmentTimeline("old")
    old = timeline.publish([_word("old", 0, 0.2)], fragment_id="old")
    reset = timeline.reset("neü")
    current = timeline.publish([_word("neü", 0, 0.3)], fragment_id="new")
    assert reset.text == "neü"
    assert reset.revision > old.revision
    assert current.revision > reset.revision
    assert [span.text for span in current.words] == ["neü"]


def test_set_text_grows_source_without_resetting_fragments_or_duration():
    timeline = AlignmentTimeline("one")
    first = timeline.publish([_word("one", 0, 0.2)], fragment_id="f")
    timeline.set_duration(0.5)
    grown = timeline.set_text("one two")

    assert grown.text == "one two"
    assert grown.revision > first.revision
    assert grown.duration == pytest.approx(0.5)
    assert [span.text for span in grown.words] == ["one"]

    final = timeline.finish()
    late = timeline.set_text("late text")
    assert late == final
    assert late.text == "one two"
    assert [span.text for span in late.words] == ["one"]

def test_wait_returns_final_and_timeout():
    timeline = AlignmentTimeline()
    with pytest.raises(TimeoutError):
        timeline.wait(0.01)

    def finish():
        time.sleep(0.02)
        timeline.finish()

    worker = threading.Thread(target=finish)
    worker.start()
    assert timeline.wait(1).final
    worker.join()


def test_callback_runs_outside_timeline_lock():
    observed = []
    holder = {}

    def on_update(snapshot):
        observed.append(snapshot.revision)
        holder["timeline"].snapshot()

    timeline = AlignmentTimeline(on_update=on_update)
    holder["timeline"] = timeline
    timeline.publish([_word("safe", 0, 0.1)], fragment_id="f")
    assert observed == [1]


def test_position_queries_do_not_dispatch_word_callbacks():
    timeline = AlignmentTimeline()
    timeline.publish(
        [
            _word("one", 0, 0.4),
            _word("two", 0.4, 0.8),
            _character("o", 0, 0.1),
        ],
        fragment_id="f",
    )
    timeline.set_duration(1)
    timeline.advance(0.5)
    assert timeline.current_word.text == "two"
    assert timeline.at(0.05, "character").text == "o"


def test_error_is_nonterminal_until_parent_finishes():
    timeline = AlignmentTimeline()
    timeline.error("model unavailable")
    assert timeline.status == "error"
    with pytest.raises(TimeoutError):
        timeline.wait(0.01)
    final = timeline.finish()
    assert final.final
    assert final.errors == ("model unavailable",)


def test_status_updates_and_errors_remain_nonterminal():
    timeline = AlignmentTimeline()
    timeline.publish([_word("kept", 0, 0.2)], fragment_id="f")
    errored = timeline.error("fallback model exceeded audio bound")
    assert errored.status == "error"
    assert errored.errors == ("fallback model exceeded audio bound",)
    disabled = timeline.set_status("disabled")
    assert disabled.status == "disabled"
    assert disabled.errors == errored.errors
    assert not disabled.final
    assert timeline.words[0].text == "kept"


def test_export_json_srt_and_vtt(tmp_path):
    timeline = AlignmentTimeline("Hi <there>! Привет.")
    timeline.publish(
        [
            _word("Hi", 0.0, 0.4, text_start=0, text_end=2),
            _word("<there>!", 0.5, 1.1, text_start=3, text_end=11),
            _character("П", 1.2, 1.3, text_start=12, text_end=13),
        ],
        fragment_id="f",
    )
    timeline.set_duration(1.5)
    snapshot = timeline.finish()
    json_path = snapshot.export(tmp_path / "alignment.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["words"][0]["word"] == "Hi"
    assert payload["characters"][0]["character"] == "П"
    srt = snapshot.export(tmp_path / "alignment.srt").read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,100" in srt
    assert "Hi <there>!" in srt
    vtt = snapshot.export(tmp_path / "alignment.vtt").read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT\n")
    assert "&lt;there&gt;!" in vtt
    assert "Привет" not in vtt


def test_project_offsets_preserves_repeats_unicode_and_unmappable_expansions():
    spans = [_word("naïve", 0, 0.2), _word("naïve", 0.2, 0.4), _word("expanded", 0.4, 0.6)]
    projected = project_text_offsets(spans, "NaÏVE naïve can't")
    assert (projected[0].text_start, projected[0].text_end) == (0, 5)
    assert (projected[1].text_start, projected[1].text_end) == (6, 11)
    assert projected[2].text_start is None
    assert projected[2].text_end is None


def test_project_offsets_uses_spoken_text_ranges():
    spans = [_word("hello", 0.0, 0.2, text_start=0, text_end=5)]
    projected = project_text_offsets(spans, "Hello world", spoken_text="hello world")
    assert (projected[0].text_start, projected[0].text_end) == (0, 5)
