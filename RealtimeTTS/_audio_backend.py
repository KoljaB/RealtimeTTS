"""Shared access to the supported PyAudio/PortAudio playback backend."""

try:
    import pyaudio as pyaudio
    import pyaudio._portaudio as pa
except ImportError as exc:
    raise ImportError(
        "RealtimeTTS local playback requires PyAudio. Install "
        "'realtimetts[playback]' and, on Linux/macOS, install PortAudio first."
    ) from exc


AUDIO_BACKEND = "pyaudio"

__all__ = ["AUDIO_BACKEND", "pa", "pyaudio"]
