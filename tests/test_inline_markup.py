from __future__ import annotations

import pytest

from RealtimeTTS.inline_markup import InlineMarkupParser


def _coalesce(actions):
    result = []
    for kind, value in actions:
        if kind == "text" and result and result[-1][0] == "text":
            result[-1] = ("text", result[-1][1] + value)
        else:
            result.append((kind, value))
    return result


def _run(parser, source: str, chunks=None):
    return _coalesce(parser.iter_actions(chunks if chunks is not None else iter(source)))


def _split_at_all_positions(source: str):
    for index in range(len(source) + 1):
        yield source[:index], source[index:]


@pytest.mark.parametrize("opener", ["[speed=.75]", "<speed .75>", "[angry]", "<angry>"])
def test_language_depth_ignores_speed_and_voice_scopes(opener):
    parser = InlineMarkupParser(
        1.0, language_tags=True, voice_tags={"angry": "A"},
    )
    _run(parser, opener * parser.MAX_LANGUAGE_DEPTH)
    actions = _run(parser, "<es>Hola</es>")
    assert [(kind, value) for kind, value in actions if kind == "language"] == [
        ("language", "es"), ("language", "en"),
    ]
    assert "".join(value for kind, value in actions if kind == "text") == "Hola"


def test_language_depth_limit_still_applies_with_mixed_scopes():
    parser = InlineMarkupParser(
        1.0, language_tags=True, voice_tags={"angry": "A"},
    )
    _run(parser, "<speed .75><angry><es>" * parser.MAX_LANGUAGE_DEPTH)
    with pytest.raises(ValueError, match="Too many nested speech language controls"):
        _run(parser, "<de>")
    assert parser.current_language == "es"

    _run(parser, "</es></angry></speed>")
    assert ("language", "de") in _run(parser, "<de>")


@pytest.mark.parametrize("opener, closer", [
    ("<speed .75>", "</speed>"),
    ("<angry>", "</angry>"),
])
def test_outer_scope_close_and_reset_restore_language_capacity(opener, closer):
    parser = InlineMarkupParser(
        1.0, language_tags=True, voice_tags={"angry": "A"},
    )
    _run(parser, opener + "<es>" * (parser.MAX_LANGUAGE_DEPTH // 2))
    _run(parser, closer)
    assert parser.current_language == "en"
    _run(parser, "<de>" * parser.MAX_LANGUAGE_DEPTH)
    with pytest.raises(ValueError, match="Too many nested speech language controls"):
        _run(parser, "<es>")

    parser.reset()
    assert parser.current_language == "en"
    _run(parser, "<es>" * parser.MAX_LANGUAGE_DEPTH)


def test_self_closing_language_controls_do_not_consume_nesting_capacity():
    parser = InlineMarkupParser(1.0, language_tags=True)
    _run(parser, "<es/>" * (2 * parser.MAX_LANGUAGE_DEPTH))
    _run(parser, "<de>" * parser.MAX_LANGUAGE_DEPTH)
    assert ("language", "fr") in _run(parser, "<fr/>")
    with pytest.raises(ValueError, match="Too many nested speech language controls"):
        _run(parser, "<en>")
    _run(parser, "</de>")
    assert ("language", "en") in _run(parser, "<en>")


def test_speed_xml_bracket_reset_percent_and_self_closing():
    parser = InlineMarkupParser(1.25, speed_tags=True)
    actions = _run(
        parser,
        '<speed factor="60%">slow</speed>[speed=reset]'
        '<sr=".8"/>fast<speed .5>inner</speed>end',
    )
    assert actions == [
        ("speed", 0.6),
        ("text", "slow"),
        ("speed", 1.25),
        ("speed", 1.25),
        ("speed", 0.8),
        ("text", "fast"),
        ("speed", 0.5),
        ("text", "inner"),
        ("speed", 0.8),
        ("text", "end"),
    ]


def test_every_chunk_boundary_has_the_same_result():
    source = "A##speed=0.5##B<speed value='2'>C</speed>D"
    expected = _run(
        InlineMarkupParser(1.0, speed_tags=True, delimiters=("##", "##")),
        source,
    )

    for left, right in _split_at_all_positions(source):
        parser = InlineMarkupParser(
            1.0, speed_tags=True, delimiters=("##", "##")
        )
        assert _run(parser, source, iter((left, right))) == expected


def test_nested_mixed_scopes_unwind_in_consistent_order():
    parser = InlineMarkupParser(
        1.0,
        speed_tags=True,
        voice_tags={"normal": "N", "angry": "A"},
        initial_voice=("normal", "N"),
    )
    actions = _run(parser, "<speed .5><angry>x<speed=2>y</angry>z</speed>")
    assert actions == [
        ("speed", 0.5),
        ("voice", ("angry", "A")),
        ("text", "x"),
        ("speed", 2.0),
        ("text", "y"),
        ("speed", 0.5),
        ("voice", ("normal", "N")),
        ("text", "z"),
        ("speed", 1.0),
    ]


def test_reset_keeps_outer_scope_for_inner_close():
    parser = InlineMarkupParser(1.0, speed_tags=True)
    actions = _run(parser, "<speed .5>a<speed reset>b</speed>c</speed>d")
    assert actions == [
        ("speed", 0.5),
        ("text", "a"),
        ("speed", 1.0),
        ("text", "b"),
        ("speed", 0.5),
        ("text", "c"),
        ("speed", 1.0),
        ("text", "d"),
    ]


def test_voice_lookup_is_exact_then_unique_casefold():
    exact = InlineMarkupParser(1.0, voice_tags={"Angry": "A", "normal": "N"})
    assert _run(exact, "[Angry]x[/Angry]") == [
        ("voice", ("Angry", "A")),
        ("text", "x"),
    ]

    ambiguous = InlineMarkupParser(
        1.0, voice_tags={"Angry": "A", "angry": "a"}
    )
    assert _run(ambiguous, "[ANGRY]x") == [
        ("text", "[ANGRY]x")
    ]


def test_generic_voice_and_registered_pauses():
    parser = InlineMarkupParser(
        1.0,
        voice_tags={"normal": "N", "angry": "A"},
        pause_tags={"pause": 0.5, "beat": 0.2},
        initial_voice=("normal", "N"),
    )
    actions = _run(
        parser,
        "<voice name='angry'>Hi</voice><style=\"normal\"/>"
        "[pause=0.2]<pause duration='0.7'/>",
    )
    assert actions == [
        ("voice", ("angry", "A")),
        ("text", "Hi"),
        ("voice", ("normal", "N")),
        ("voice", ("normal", "N")),
        ("pause", ("pause", 0.2)),
        ("pause", ("pause", 0.7)),
    ]


def test_unknown_invalid_quoted_and_unfinished_markup_stays_literal():
    oversized = "[" + ("x" * 300) + "]"
    source = (
        'u<unknown title="a > b">v</unknown>'
        '<speed factor="bad">w'
        + oversized
        + 'tail<voice name="angry"'
    )
    parser = InlineMarkupParser(
        1.0, speed_tags=True, voice_tags={"angry": "A"}
    )
    actions = _run(parser, source)
    assert actions == [("text", source)]


def test_configured_bracket_delimiter_wins_for_shared_opener():
    parser = InlineMarkupParser(
        1.0, speed_tags=True, delimiters=("[", "}}")
    )
    source = "A[speed=.5}}B[speed=2]C"
    assert _run(parser, source) == [
        ("text", "A"),
        ("speed", 0.5),
        ("text", "B[speed=2]C"),
    ]


def test_configured_angle_delimiter_wins_for_shared_opener():
    parser = InlineMarkupParser(
        1.0, speed_tags=True, delimiters=("<", ">>")
    )
    source = "A<speed=.5>>B<speed=2>C"
    assert _run(parser, source) == [
        ("text", "A"),
        ("speed", 0.5),
        ("text", "B<speed=2>C"),
    ]


def test_oversized_candidate_is_literal_at_final_close_character():
    tag = "v" * 255
    source = f"[{tag}]"
    parser = InlineMarkupParser(1.0, voice_tags={tag: "V"})
    assert _run(parser, source) == [("text", source)]


def test_canonical_speed_attribute_without_spacing():
    parser = InlineMarkupParser(1.0, speed_tags=True)
    assert _run(parser, '<speed factor=".6">x</speed>') == [
        ("speed", 0.6),
        ("text", "x"),
        ("speed", 1.0),
    ]


def test_builtin_and_custom_delimiters_work_together_and_pause_override():
    parser = InlineMarkupParser(
        1.0,
        speed_tags=True,
        pause_tags={"pause": 0.4},
        delimiters=("##", "##"),
    )
    actions = _run(parser, "A##speed=2##B<speed .5>C[pause]D[pause=0.2]E")
    assert actions == [
        ("text", "A"),
        ("speed", 2.0),
        ("text", "B"),
        ("speed", 0.5),
        ("text", "C"),
        ("pause", ("pause", 0.4)),
        ("text", "D"),
        ("pause", ("pause", 0.2)),
        ("text", "E"),
    ]


def test_initial_unknown_voice_restoration_is_suppressed():
    parser = InlineMarkupParser(
        1.0, voice_tags={"angry": "A"}, initial_voice=("normal", "N")
    )
    assert _run(parser, "<angry>x</angry>tail") == [
        ("voice", ("angry", "A")),
        ("text", "xtail"),
    ]


def test_char_by_char_stream_and_known_unmatched_close():
    parser = InlineMarkupParser(
        1.0, speed_tags=True, voice_tags={"angry": "A"}
    )
    source = "x</speed>y<angry/>z[/speed]q"
    assert _run(parser, source, (char for char in source)) == [
        ("text", "xy"),
        ("voice", ("angry", "A")),
        ("text", "zq"),
    ]


def test_none_tag_initial_voice_is_a_known_restore_baseline():
    parser = InlineMarkupParser(
        1.0, voice_tags={"angry": "A"}, initial_voice=(None, "engine")
    )
    assert _run(parser, "<angry>x</angry>") == [
        ("voice", ("angry", "A")),
        ("text", "x"),
        ("voice", (None, "engine")),
    ]


def test_state_persists_between_calls_until_reset():
    parser = InlineMarkupParser(
        1.0,
        speed_tags=True,
        voice_tags={"normal": "N", "angry": "A"},
        initial_voice=("normal", "N"),
    )
    assert _run(parser, "<speed .5><angry>one") == [
        ("speed", 0.5),
        ("voice", ("angry", "A")),
        ("text", "one"),
    ]
    assert _run(parser, "two</angry></speed>") == [
        ("text", "two"),
        ("voice", ("normal", "N")),
        ("speed", 1.0),
    ]
    parser.reset()
    assert _run(parser, "<speed .5>fresh</speed>") == [
        ("speed", 0.5),
        ("text", "fresh"),
        ("speed", 1.0),
    ]


def test_arbitrary_registered_tags_and_pause_value_attribute():
    parser = InlineMarkupParser(
        1.0,
        voice_tags={"very happy": "V", "fröhlich": "F"},
        pause_tags={"short pause": 0.5},
    )
    assert _run(parser, "[very happy]x[/very happy]") == [
        ("voice", ("very happy", "V")),
        ("text", "x"),
    ]
    assert _run(parser, "[fröhlich]") == [
        ("voice", ("fröhlich", "F")),
    ]
    assert _run(parser, "[short pause: 0.2]") == [
        ("pause", ("short pause", 0.2)),
    ]
    assert _run(parser, "<short pause value=.2/>") == [
        ("pause", ("short pause", 0.2)),
    ]


def test_resolve_tag_returns_actions_or_none_and_preserves_scope_state():
    parser = InlineMarkupParser(1.0, speed_tags=True)
    assert parser.resolve_tag("speed=.5") == [("speed", 0.5)]
    assert parser.resolve_tag("/speed") == [("speed", 1.0)]
    assert parser.resolve_tag("unknown") is None
    assert parser.resolve_tag("/speed") == []
