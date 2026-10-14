"""Focused tests for the thread-safe tempo timeline."""

from dataclasses import FrozenInstanceError

import pytest

from RealtimeTTS.tempo_timeline import TempoTimeline


def test_multirate_offsets_and_actual_closed_endpoints():
    timeline = TempoTimeline()
    timeline.add_input(1.0)
    timeline.add_output(1.0)
    timeline.finish_segment(next_speed=2.0)
    timeline.add_input(2.0)
    timeline.add_output(1.0)
    timeline.finish_segment()

    segments = timeline.snapshot()
    assert len(segments) == 2
    assert segments[0].source_start == 0.0
    assert segments[0].source_end == 1.0
    assert segments[0].output_start == 0.0
    assert segments[0].output_end == 1.0
    assert segments[0].closed
    assert segments[1].speed == 2.0
    assert segments[1].source_start == 1.0
    assert segments[1].source_end == 3.0
    assert segments[1].output_start == 1.0
    assert segments[1].output_end == 2.0
    assert timeline.source_to_output(2.0) == pytest.approx(1.5)
    assert timeline.output_to_source(1.5) == pytest.approx(2.0)


def test_short_duration_closed_segment_anchors_actual_output_end():
    timeline = TempoTimeline(2.0)
    timeline.add_input(0.003)
    timeline.add_output(0.0017)
    timeline.finish_segment()
    segment = timeline.snapshot()[0]
    assert timeline.source_to_output(segment.source_end) == pytest.approx(0.0017)
    assert timeline.output_to_source(segment.output_end) == pytest.approx(0.003)
    assert timeline.source_to_output(0.0015) == pytest.approx(0.0017 * 0.5)


def test_closed_segments_are_immutable_and_never_rewritten():
    timeline = TempoTimeline()
    timeline.add_input(1.0)
    timeline.add_output(0.5)
    timeline.finish_segment(next_speed=1.5)
    old = timeline.snapshot()[0]
    old_mapping = timeline.source_to_output(0.5)
    with pytest.raises(FrozenInstanceError):
        old.output_end = 99.0
    timeline.add_input(0.25)
    timeline.add_output(0.4)
    timeline.finish_segment()
    assert timeline.snapshot()[0] == old
    assert timeline.source_to_output(0.5) == pytest.approx(old_mapping)


def test_open_and_closed_mappings_are_monotonic_with_future_extrapolation():
    timeline = TempoTimeline(1.25)
    timeline.add_input(0.4)
    timeline.add_output(0.25)
    source_values = [timeline.source_to_output(value) for value in (0.0, 0.1, 0.4, 1.0)]
    output_values = [timeline.output_to_source(value) for value in (0.0, 0.1, 0.25, 0.5, 1.0)]
    assert source_values == sorted(source_values)
    assert output_values == sorted(output_values)

    timeline.finish_segment(next_speed=0.5)
    timeline.add_input(0.2)
    timeline.add_output(0.5)
    timeline.finish_segment()
    source_values = [timeline.source_to_output(value) for value in (0.0, 0.4, 0.6)]
    output_values = [timeline.output_to_source(value) for value in (0.0, 0.25, 0.75)]
    assert source_values == sorted(source_values)
    assert output_values == sorted(output_values)


def test_identity_speed_one_is_exact_and_changed_tracks_any_segment():
    timeline = TempoTimeline()
    timeline.add_input(1.25)
    timeline.add_output(1.25)
    assert timeline.source_to_output(0.75) == 0.75
    assert timeline.output_to_source(0.75) == 0.75
    assert not timeline.changed
    timeline.begin(1.2)
    assert timeline.changed
    timeline.add_input(0.1)
    timeline.add_output(0.08)
    timeline.finish_segment()
    assert timeline.changed


def test_reset_starts_a_new_generation_and_discards_history():
    timeline = TempoTimeline(1.5)
    timeline.add_input(1.0)
    timeline.add_output(0.5)
    timeline.finish_segment()
    generation = timeline.generation
    timeline.reset(1.0)

    segments = timeline.snapshot()
    assert timeline.generation == generation + 1
    assert len(segments) == 1
    segment = segments[0]
    assert segment.source_start == 0.0
    assert segment.output_start == 0.0
    assert segment.source_end == 0.0
    assert segment.output_end == 0.0
    assert segment.speed == 1.0
    assert not segment.closed
    assert not timeline.changed
    assert timeline.source_duration == 0.0
    assert timeline.output_duration == 0.0
