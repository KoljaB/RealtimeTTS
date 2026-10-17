import inspect

from RealtimeTTS import TextToAudioStream


def test_mixed_nltk_rule_based_tokenizer_is_the_stream_default():
    assert (
        inspect.signature(TextToAudioStream).parameters["tokenizer"].default
        == "nltk+rule-based"
    )


def test_play_inherits_the_tokenizer_configured_on_the_stream():
    assert inspect.signature(TextToAudioStream.play).parameters["tokenizer"].default == ""
    assert (
        inspect.signature(TextToAudioStream.play_async).parameters["tokenizer"].default
        == ""
    )


def test_play_inherits_the_language_configured_on_the_stream():
    assert inspect.signature(TextToAudioStream.play).parameters["language"].default == ""
    assert inspect.signature(TextToAudioStream.play_async).parameters["language"].default == ""


def test_fragment_lookahead_is_opt_in_on_sync_and_async_playback():
    assert (
        inspect.signature(TextToAudioStream.play).parameters[
            "fragment_lookahead_words"
        ].default
        == 0
    )
    assert inspect.signature(TextToAudioStream.play_async).parameters[
        "fragment_lookahead_words"
    ].default == 0
