"""Collection policy for the release test suite.

Files that predate the pytest suite are runnable examples, not tests.  They
must not be imported during collection: some import optional SDKs and one
starts playback as soon as it is imported.
"""

collect_ignore = [
    "test_callbacks.py",
    "test_on_audio_chunk_callback.py",
]
