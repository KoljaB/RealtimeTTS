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
