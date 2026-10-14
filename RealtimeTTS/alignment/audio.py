from __future__ import annotations

import io
import os
import wave
from typing import Optional, Tuple

import numpy as np

from .. import _audio_formats as formats


_PCM_FORMATS = {
    formats.FLOAT32: (np.float32, 4, 1.0),
    formats.INT16: (np.int16, 2, 32768.0),
    formats.INT32: (np.int32, 4, 2147483648.0),
    formats.INT8: (np.int8, 1, 128.0),
    formats.UINT8: (np.uint8, 1, 128.0),
}
_CUSTOM_FORMAT = formats.CUSTOM


def is_alignable_stream_format(
    stream_format: int,
    channels: int,
    sample_rate: int,
) -> bool:
    if channels <= 0 or sample_rate <= 0:
        return False
    return stream_format in _PCM_FORMATS or stream_format == _CUSTOM_FORMAT


def pcm_chunk_duration_seconds(
    audio_bytes: bytes,
    stream_format: int,
    channels: int,
    sample_rate: int,
) -> Optional[float]:
    if channels <= 0 or sample_rate <= 0:
        return None
    format_info = _PCM_FORMATS.get(stream_format)
    if format_info is None:
        return None
    _, sample_width, _ = format_info
    frame_count = len(audio_bytes) // (sample_width * channels)
    return frame_count / float(sample_rate)


def audio_bytes_to_mono_float32(
    audio_bytes: bytes,
    stream_format: int,
    channels: int,
    sample_rate: int,
) -> Tuple[np.ndarray, int]:
    if not audio_bytes:
        return np.array([], dtype=np.float32), sample_rate

    if stream_format == _CUSTOM_FORMAT:
        return _decode_mp3_to_mono_float32(audio_bytes)

    format_info = _PCM_FORMATS.get(stream_format)
    if format_info is None:
        raise ValueError(f"Unsupported audio stream format for alignment: {stream_format}")

    dtype, _, scale = format_info
    audio = np.frombuffer(audio_bytes, dtype=dtype)
    if channels > 1:
        usable_samples = (len(audio) // channels) * channels
        audio = audio[:usable_samples].reshape((-1, channels)).mean(axis=1)

    if dtype == np.float32:
        audio_float = audio.astype(np.float32, copy=False)
    elif dtype == np.uint8:
        audio_float = (audio.astype(np.float32) - 128.0) / scale
    else:
        audio_float = audio.astype(np.float32) / scale

    return np.clip(audio_float, -1.0, 1.0), sample_rate


def wav_file_to_mono_float32(
    wav_path: str | os.PathLike,
    channel: int | None = None,
) -> Tuple[np.ndarray, int]:
    with wave.open(os.fspath(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        audio_bytes = wav_file.readframes(wav_file.getnframes())

    if channels <= 0 or sample_rate <= 0:
        raise ValueError("WAV file must have a positive channel count and sample rate")
    if not audio_bytes:
        return np.array([], dtype=np.float32), sample_rate

    audio = _pcm_wav_bytes_to_float32(audio_bytes, sample_width)
    if channels > 1:
        usable_samples = (len(audio) // channels) * channels
        audio = audio[:usable_samples].reshape((-1, channels))
        if channel is None:
            audio = audio.mean(axis=1)
        elif 0 <= channel < channels:
            audio = audio[:, channel]
        else:
            raise ValueError(
                f"WAV channel index {channel} is outside available channels 0-{channels - 1}"
            )
    elif channel not in (None, 0):
        raise ValueError("Mono WAV files only support channel 0")

    return np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0), sample_rate


def compressed_audio_duration_seconds(audio_bytes: bytes) -> float:
    audio, sample_rate = _decode_mp3_to_mono_float32(audio_bytes)
    if sample_rate <= 0:
        return 0.0
    return len(audio) / float(sample_rate)


def _pcm_wav_bytes_to_float32(audio_bytes: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        samples = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32)
        return (samples - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        bytes_24 = np.frombuffer(audio_bytes, dtype=np.uint8).reshape((-1, 3))
        samples = (
            bytes_24[:, 0].astype(np.int32)
            | (bytes_24[:, 1].astype(np.int32) << 8)
            | (bytes_24[:, 2].astype(np.int32) << 16)
        )
        samples = np.where(samples & 0x800000, samples | ~0xFFFFFF, samples)
        return samples.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return (
            np.frombuffer(audio_bytes, dtype="<i4").astype(np.float32)
            / 2147483648.0
        )
    raise ValueError(f"Unsupported PCM WAV sample width: {sample_width}")


def _decode_mp3_to_mono_float32(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    from pydub import AudioSegment

    segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    samples = np.array(segment.get_array_of_samples())
    channels = max(segment.channels, 1)
    if channels > 1:
        samples = samples.reshape((-1, channels)).mean(axis=1)

    scale = float(1 << (8 * segment.sample_width - 1))
    audio = (samples.astype(np.float32) / scale).clip(-1.0, 1.0)
    return audio, int(segment.frame_rate)
