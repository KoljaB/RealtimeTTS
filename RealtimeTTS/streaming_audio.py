import threading
from typing import Any

from ._audio_backend import pyaudio


class DecodedAudioQueue:
    """Decode compressed MP3 chunks before forwarding float32 mono PCM."""

    def __init__(self, wrapped: Any, sample_rate: int = 24_000) -> None:
        self.wrapped = wrapped
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.output_format = (pyaudio.paFloat32, 1, self.sample_rate)
        self._lock = threading.Lock()
        self._active = False
        self.begin()

    def begin(self) -> None:
        with self._lock:
            if "_decoder" in self.__dict__:
                self._decoder.reset()
            else:
                from .alignment.decoder import MP3Decoder

                self._decoder = MP3Decoder(self.sample_rate)
            self._active = True

    def put(self, item: Any, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            if not self._active:
                return
            for chunk in self._decoder.feed(bytes(item)):
                if not self._active:
                    return
                self.wrapped.put(chunk, *args, **kwargs)

    def put_nowait(self, item: Any) -> None:
        self.put(item, block=False)

    def finish(self) -> None:
        with self._lock:
            if not self._active:
                return
            for chunk in self._decoder.finish():
                if not self._active:
                    return
                self.wrapped.put(chunk)
            self._active = False

    def cancel(self) -> None:
        with self._lock:
            self._active = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)
