"""PortAudio sample-format ABI values, usable without a playback installation.

Engines expose these integer formats in get_stream_info(). Decoding or sending
PCM does not require opening a device or importing PyAudio.
"""

FLOAT32 = 0x00000001
INT32 = 0x00000002
INT24 = 0x00000004
INT16 = 0x00000008
INT8 = 0x00000010
UINT8 = 0x00000020
CUSTOM = 0x00010000
