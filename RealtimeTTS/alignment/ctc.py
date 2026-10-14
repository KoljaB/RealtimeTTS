from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .base import CharacterAlignment, WordAlignment


@dataclass(frozen=True)
class TokenSpan:
    token_index: int
    token_id: int
    start_frame: int
    end_frame: int
    score: float


def log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    max_logits = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logits
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def ctc_forced_align(
    log_probs: np.ndarray,
    targets: Sequence[int],
    blank_id: int = 0,
    nonblank_frame_mask: Sequence[bool] | np.ndarray | None = None,
    nonblank_forbidden_penalty: float | None = -30.0,
) -> list[TokenSpan]:
    """
    Viterbi-align a known token sequence against frame-level CTC log-probs.
    """

    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape [frames, vocab]")

    targets = [int(token_id) for token_id in targets]
    if not targets:
        return []

    frame_count, vocab_size = log_probs.shape
    nonblank_frame_mask = _validate_nonblank_frame_mask(
        nonblank_frame_mask,
        frame_count,
    )
    if frame_count == 0:
        raise ValueError("log_probs must contain at least one frame")
    if min(targets) < 0 or max(targets) >= vocab_size:
        raise ValueError("target token id is outside the logits vocabulary")

    states = np.asarray(_ctc_states(targets, blank_id), dtype=np.int32)
    state_count = len(states)
    scores = np.full(state_count, -np.inf, dtype=np.float64)
    scores[0] = 0.0
    backptr = np.zeros((frame_count, state_count), dtype=np.uint8)
    columns = np.arange(state_count)
    skip = np.zeros(state_count, dtype=bool)
    skip[2:] = (states[2:] != blank_id) & (states[2:] != states[:-2])
    for frame in range(frame_count):
        one = np.r_[-np.inf, scores[:-1]]
        two = np.r_[-np.inf, -np.inf, scores[:-2]]
        two[~skip] = -np.inf
        candidates = np.stack((scores, one, two))
        moves = candidates.argmax(axis=0)
        emission = log_probs[frame, states]
        if nonblank_frame_mask is not None and not nonblank_frame_mask[frame]:
            penalty = -np.inf if nonblank_forbidden_penalty is None else float(nonblank_forbidden_penalty)
            emission = emission + np.where(states == blank_id, 0.0, penalty)
        scores = candidates[moves, columns] + emission
        backptr[frame] = moves

    final_states = (state_count - 1, state_count - 2)
    final_state = max(final_states, key=lambda state: scores[state])
    if not np.isfinite(scores[final_state]):
        raise ValueError("no valid CTC alignment path found")
    path = [final_state] * frame_count
    state = final_state
    for frame in range(frame_count - 1, -1, -1):
        path[frame] = state
        state -= int(backptr[frame, state])
    return _path_to_token_spans(path, states, targets, log_probs)


def token_spans_to_word_alignments(
    token_spans: Sequence[TokenSpan],
    words: Sequence[str],
    word_token_ranges: Sequence[tuple[int, int]],
    frame_duration_seconds: float,
) -> list[WordAlignment]:
    alignments: list[WordAlignment] = []
    span_by_index = {span.token_index: span for span in token_spans}

    for word, (start_index, end_index) in zip(words, word_token_ranges):
        spans = [
            span_by_index[index]
            for index in range(start_index, end_index)
            if index in span_by_index
        ]
        if not spans:
            continue

        start_frame = min(span.start_frame for span in spans)
        end_frame = max(span.end_frame for span in spans)
        alignments.append(
            WordAlignment(
                word=word,
                start_time=start_frame * frame_duration_seconds,
                end_time=end_frame * frame_duration_seconds,
            )
        )

    return alignments


def token_spans_to_character_alignments(
    token_spans: Sequence[TokenSpan],
    characters: Sequence[str],
    character_token_ranges: Sequence[tuple[int, int]],
    frame_duration_seconds: float,
) -> list[CharacterAlignment]:
    alignments: list[CharacterAlignment] = []
    span_by_index = {span.token_index: span for span in token_spans}

    for index, (character, (start_index, end_index)) in enumerate(
        zip(characters, character_token_ranges)
    ):
        spans = [
            span_by_index[token_index]
            for token_index in range(start_index, end_index)
            if token_index in span_by_index
        ]
        if not spans:
            continue

        start_frame = min(span.start_frame for span in spans)
        end_frame = max(span.end_frame for span in spans)
        score = float(np.mean([span.score for span in spans]))
        alignments.append(
            CharacterAlignment(
                character=character,
                start_time=start_frame * frame_duration_seconds,
                end_time=end_frame * frame_duration_seconds,
                index=index,
                score=score,
            )
        )

    return alignments


def _ctc_states(targets: Sequence[int], blank_id: int) -> list[int]:
    states = [blank_id]
    for target in targets:
        states.append(int(target))
        states.append(blank_id)
    return states


def _validate_nonblank_frame_mask(
    nonblank_frame_mask: Sequence[bool] | np.ndarray | None,
    frame_count: int,
) -> np.ndarray | None:
    if nonblank_frame_mask is None:
        return None
    mask = np.asarray(nonblank_frame_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != frame_count:
        raise ValueError("nonblank_frame_mask must have one boolean per frame")
    return mask


def _emission_score(
    log_probs: np.ndarray,
    frame: int,
    token_id: int,
    blank_id: int,
    nonblank_frame_mask: np.ndarray | None,
    nonblank_forbidden_penalty: float | None,
) -> float:
    score = float(log_probs[frame, token_id])
    if (
        nonblank_frame_mask is not None
        and token_id != blank_id
        and not bool(nonblank_frame_mask[frame])
    ):
        if nonblank_forbidden_penalty is None:
            return -np.inf
        return score + float(nonblank_forbidden_penalty)
    return score


def _backtrack_path(backptr: np.ndarray, final_state: int) -> list[int]:
    path = [final_state] * backptr.shape[0]
    state = final_state
    for frame in range(backptr.shape[0] - 1, 0, -1):
        path[frame] = state
        state = int(backptr[frame, state])
        if state < 0:
            raise ValueError("invalid CTC backpointer path")
    path[0] = state
    return path


def _path_to_token_spans(
    path: Sequence[int],
    states: Sequence[int],
    targets: Sequence[int],
    log_probs: np.ndarray,
) -> list[TokenSpan]:
    frames_by_token_index: dict[int, list[int]] = {
        token_index: [] for token_index in range(len(targets))
    }
    for frame, state in enumerate(path):
        if state % 2 == 1:
            frames_by_token_index[state // 2].append(frame)

    spans: list[TokenSpan] = []
    for token_index, token_id in enumerate(targets):
        frames = frames_by_token_index[token_index]
        if not frames:
            raise ValueError(f"target token {token_index} was not aligned")

        start_frame = min(frames)
        end_frame = max(frames) + 1
        frame_scores = [log_probs[frame, token_id] for frame in frames]
        spans.append(
            TokenSpan(
                token_index=token_index,
                token_id=int(token_id),
                start_frame=start_frame,
                end_frame=end_frame,
                score=float(np.mean(frame_scores)),
            )
        )

    return spans
