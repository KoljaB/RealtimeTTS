"""Incremental compressed-audio decoding for the common stream clock."""
from __future__ import annotations

import numpy as np


class MP3Decoder:
    """Decode arbitrary MP3 packet boundaries to mono float32 PCM.

    Decoding happens once, before both playback and alignment. The two consumers
    therefore use exactly the same samples, including codec delay and padding.
    """

    def __init__(self, sample_rate=24000):
        try:
            import av
        except ImportError as exc:
            raise ImportError(
                "Streaming MP3 audio requires PyAV. "
                "Install realtimetts[openai] or pip install av."
            ) from exc
        self.sample_rate = int(sample_rate)
        self._av = av
        self.reset()

    def reset(self):
        self._codec = self._av.CodecContext.create("mp3", "r")
        self._resampler = self._av.AudioResampler(
            format="flt", layout="mono", rate=self.sample_rate
        )
        self._header = bytearray()
        self._header_done = False
        self._skip = None
        self._finished = False

    def _strip_id3(self, data):
        # ID3 tags can themselves span many network chunks. They are container
        # metadata, not MP3 codec packets; do not feed them to the raw parser.
        if self._header_done:
            return data
        self._header.extend(data)
        if self._skip is not None:
            count = min(self._skip, len(self._header))
            del self._header[:count]
            self._skip -= count
            if self._skip:
                return b""
            self._header_done = True
        elif len(self._header) < 3:
            return b""
        if self._header_done:
            pass
        elif self._header[:3] != b"ID3":
            self._header_done = True
        else:
            if len(self._header) < 10:
                return b""
            if self._skip is None:
                size = 0
                for byte in self._header[6:10]:
                    if byte & 0x80:
                        raise ValueError("Invalid MP3 ID3 length")
                    size = (size << 7) | byte
                self._skip = 10 + size + (10 if self._header[5] & 0x10 else 0)
                if self._skip > 16 * 1024 * 1024:
                    raise ValueError("MP3 metadata exceeds 16 MiB")
            count = min(self._skip, len(self._header))
            del self._header[:count]
            self._skip -= count
            if self._skip:
                return b""
            self._header_done = True
        result = bytes(self._header)
        self._header.clear()
        return result

    def _frames(self, frames):
        for frame in frames:
            # Some providers reset timestamps between packets/fragments.
            frame.pts = None
            for output in self._resampler.resample(frame):
                yield output.to_ndarray().reshape(-1).astype("<f4", copy=False).tobytes()

    def feed(self, data):
        if self._finished:
            raise RuntimeError("MP3 decoder has already been finished")
        data = self._strip_id3(bytes(data))
        if data:
            for packet in self._codec.parse(data):
                yield from self._frames(self._codec.decode(packet))

    def finish(self):
        if self._finished:
            return
        self._finished = True
        for packet in self._codec.parse(b""):
            yield from self._frames(self._codec.decode(packet))
        yield from self._frames(self._codec.decode(None))
        for output in self._resampler.resample(None):
            yield output.to_ndarray().reshape(-1).astype("<f4", copy=False).tobytes()
