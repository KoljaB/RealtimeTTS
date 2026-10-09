from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class WordAlignment:
    word: str
    start_time: float
    end_time: float
    score: Optional[float] = None


@dataclass(frozen=True)
class CharacterAlignment:
    character: str
    start_time: float
    end_time: float
    index: int
    score: Optional[float] = None


@dataclass(frozen=True)
class SpeechSegment:
    start_time: float
    end_time: float


class WordAligner(Protocol):
    def align_words(
        self,
        text: str,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Sequence[WordAlignment]:
        """
        Align known text to generated audio.

        Args:
            text: The exact text used to synthesize the audio.
            audio: Mono float32 PCM audio in the range [-1.0, 1.0].
            sample_rate: Audio sample rate in Hz.

        Returns:
            Word alignments with times relative to the start of audio.
        """


class CharacterAligner(Protocol):
    def align_characters(
        self,
        text: str,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Sequence[CharacterAlignment]:
        """
        Align known text to generated audio.

        Args:
            text: The exact text used to synthesize the audio.
            audio: Mono float32 PCM audio in the range [-1.0, 1.0].
            sample_rate: Audio sample rate in Hz.

        Returns:
            Character alignments with times relative to the start of audio.
        """


class VoiceActivityDetector(Protocol):
    def detect_speech(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Sequence[SpeechSegment]:
        """
        Detect speech regions in mono float32 PCM audio.

        Args:
            audio: Mono float32 PCM audio in the range [-1.0, 1.0].
            sample_rate: Audio sample rate in Hz.

        Returns:
            Speech segments with times relative to the start of audio.
        """
