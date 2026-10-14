"""Gap ownership must follow spoken text across language/tempo boundaries."""
import pytest

from RealtimeTTS.speech_boundaries import SpeechBoundaryPauses


def emit(gaps, text, language):
    duration = gaps.after(text, language)
    gaps.explicit_pause(duration)
    return duration


def test_no_first_audio_delay_and_punctuationless_switch():
    gaps = SpeechBoundaryPauses()
    assert gaps.before("Hello", "en") == ("Hello", 0)
    assert emit(gaps, "Hello", "en") == 0
    assert gaps.before("hola", "es") == ("hola", .08)
    assert emit(gaps, "hola", "es") == 0
    assert gaps.before(" again", "en") == ("again", .08)


@pytest.mark.parametrize("mark,duration", [(",", .15), (".", .30)])
def test_punctuation_outside_language_scope_belongs_to_previous_audio(mark, duration):
    gaps = SpeechBoundaryPauses()
    emit(gaps, "el árbol", "es")
    assert gaps.before(mark + " Try it.", "en") == ("Try it.", duration)


def test_punctuation_only_fragment_preserves_gap_without_duplicate():
    gaps = SpeechBoundaryPauses()
    emit(gaps, "el árbol", "es")
    assert gaps.before(".", "en") == ("", .30)
    gaps.explicit_pause(.30)
    assert gaps.before("Try it.", "en") == ("Try it.", 0)


@pytest.mark.parametrize("text,duration", [("hola,", .15), ("hola.", .30)])
def test_existing_pause_covers_language_transition(text, duration):
    gaps = SpeechBoundaryPauses()
    assert emit(gaps, text, "es") == duration
    assert gaps.before("Try it.", "en") == ("Try it.", 0)


def test_late_period_tops_up_comma_pause():
    gaps = SpeechBoundaryPauses()
    emit(gaps, "hola,", "es")
    assert gaps.before(". Try it.", "en") == ("Try it.", .15)


def test_same_language_and_delivery_tags_unchanged():
    gaps = SpeechBoundaryPauses()
    emit(gaps, "Hello", "en")
    assert gaps.before("[warm] Again.", "en") == ("[warm] Again.", 0)
    assert gaps.before(".NET is a framework", "en") == (".NET is a framework", 0)


def test_language_switch_preserves_leading_period_in_word():
    gaps = SpeechBoundaryPauses()
    emit(gaps, "hola", "es")
    assert gaps.before(".NET is a framework", "en") == (".NET is a framework", .08)


@pytest.mark.parametrize("invalid", [-1, float("inf"), float("nan")])
def test_invalid_gaps_rejected(invalid):
    with pytest.raises(ValueError):
        SpeechBoundaryPauses(transition=invalid)


def test_unsent_gap_does_not_hide_a_later_transition():
    gaps = SpeechBoundaryPauses()
    gaps.after("hola", "es")
    assert gaps.before("Skipped", "en")[1] == .08
    assert gaps.before("Actually spoken", "en")[1] == .08


def test_closing_delimiters_move_with_previous_phrase_opening_tags_stay():
    gaps = SpeechBoundaryPauses()
    emit(gaps, "hola", "es")
    assert gaps.before("). Next.", "en") == ("Next.", .30)
    assert gaps.before("[warm] Next.", "en") == ("[warm] Next.", .08)
