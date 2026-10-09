"""Incremental CTC timestamps and a completion-based TTS tail cutoff.

The audio producer only copies into a bounded buffer. One CPU worker owns
alignment; incomplete text remains incomplete instead of being forced to fit.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import logging
import re
import threading
import time
import numpy as np

from .base import CharacterAlignment, WordAlignment


MIN_COMPLETION_TAIL_SECONDS = 0.25


class CTCPrefixTracker:
    """Match an unconstrained CTC decoding to the expected text prefix."""

    def __init__(self, text, targets, blank_id, *, minimum_probability=0.1,
                 separator_id=None):
        self.text = text
        self.targets = np.asarray(targets, dtype=np.int32)
        self.blank_id = blank_id
        self.minimum_probability = minimum_probability
        self.separators = {int(t) for c, t in zip(text, targets) if c.isspace()}
        if separator_id is not None:
            self.separators.add(int(separator_id))
        self.words = list(re.finditer(r"\S+", text))
        self.runs = []
        self.previous = None

    def advance(self, log_probs, times):
        for row, timestamp in zip(log_probs, times):
            token = int(np.argmax(row))
            probability = float(np.exp(row[token]))
            if token == self.previous and self.runs:
                self.runs[-1][2] = float(timestamp)
                self.runs[-1][3] = max(self.runs[-1][3], probability)
            else:
                self.runs.append([token, float(timestamp), float(timestamp), probability])
            self.previous = token

    def result(self, frame_seconds):
        speech = [r for r in self.runs
                  if r[0] != self.blank_id and r[0] not in self.separators]
        expected = [(i, c, int(t)) for i, (c, t) in enumerate(zip(self.text, self.targets))
                    if not c.isspace()]
        characters = []
        for (token, start, end, probability), (index, character, target) in zip(speech, expected):
            if token != target or probability < self.minimum_probability:
                break
            characters.append(CharacterAlignment(character, start, end + frame_seconds,
                                                  index, probability))
        complete = bool(expected) and len(characters) == len(expected) == len(speech)
        by_index = {c.index: c for c in characters}
        # Word separators are formatting, not mandatory spoken characters.
        for index, character in enumerate(self.text):
            previous = by_index.get(index - 1)
            following = by_index.get(index + 1)
            if not character.isspace() or previous is None:
                continue
            if following is not None:
                by_index[index] = CharacterAlignment(
                    character, previous.end_time, max(previous.end_time, following.start_time),
                    index, min(previous.score, following.score))
            else:
                separator = next((r for r in self.runs
                                  if r[0] in self.separators and r[1] >= previous.end_time), None)
                if separator is not None:
                    by_index[index] = CharacterAlignment(
                        character, separator[1], separator[2] + frame_seconds, index, separator[3])
        characters = sorted(by_index.values(), key=lambda c: c.index)
        words = []
        for match in self.words:
            covered = [by_index.get(i) for i in range(match.start(), match.end())]
            if all(c is not None for c in covered):
                words.append(WordAlignment(match.group(), covered[0].start_time,
                                           covered[-1].end_time, min(c.score for c in covered)))
        return characters, words, complete

def display_ctc_characters(log_probs, targets, text, times, blank_id, frame_seconds):
    """Align the known text prefix without forcing an unspoken suffix."""
    if not len(log_probs) or not len(targets):
        return []
    states = np.full(len(targets) * 2 + 1, blank_id, dtype=np.int32)
    states[1::2] = targets
    skip = np.zeros(len(states), dtype=bool)
    skip[2:] = (states[2:] != blank_id) & (states[2:] != states[:-2])
    scores = np.full(len(states), -np.inf)
    scores[0] = 0
    back = np.zeros((len(log_probs), len(states)), dtype=np.uint8)
    columns = np.arange(len(states))
    for frame, probabilities in enumerate(log_probs):
        one = np.r_[-np.inf, scores[:-1]]
        two = np.r_[-np.inf, -np.inf, scores[:-2]]
        two[~skip] = -np.inf
        candidates = np.stack((scores, one, two))
        moves = candidates.argmax(axis=0)
        scores = candidates[moves, columns] + probabilities[states]
        back[frame] = moves
    state = int(scores.argmax())
    path = np.empty(len(log_probs), dtype=np.int32)
    for frame in range(len(log_probs) - 1, -1, -1):
        path[frame] = state
        state -= int(back[frame, state])
    characters = []
    for index in range((int(path.max()) + 1) // 2):
        frames = np.flatnonzero(path == index * 2 + 1)
        if frames.size:
            characters.append(CharacterAlignment(
                text[index], float(times[frames[0]]),
                float(times[frames[-1]] + frame_seconds), index,
                float(np.exp(log_probs[frames, targets[index]]).mean())))
    return characters


class StreamingTextAligner:
    """Shared CPU backend; independent alignment state for each TTS fragment."""

    def __init__(self, model_dir=None, *, backend=None, threads=2, language=None,
                 multilingual_model=None, multilingual_backend=None,
                 interval_seconds=0.16, window_seconds=12.0, tail_seconds=0.2,
                 stall_seconds=None):
        if multilingual_model is not None:
            from .onnx_parakeet import OnnxParakeetBackend
            multilingual_backend = OnnxParakeetBackend(multilingual_model, threads=threads)
        if backend is None and model_dir is not None:
            from .onnx_ctc import OnnxCTCBackend
            backend = OnnxCTCBackend(model_dir, threads=threads, language=language)
        if backend is None:
            backend = multilingual_backend
        if backend is None:
            raise ValueError("An alignment model or backend is required")
        self.multilingual_backend = multilingual_backend
        if interval_seconds <= 0 or window_seconds <= interval_seconds or tail_seconds <= 0:
            raise ValueError("Invalid streaming alignment intervals")
        self.backend = backend
        self.interval_seconds = float(interval_seconds)
        self.window_seconds = float(window_seconds)
        # Keep at least a quarter second after verified text for final consonants.
        self.tail_seconds = max(float(tail_seconds), MIN_COMPLETION_TAIL_SECONDS)
        self.stall_seconds = float(stall_seconds) if stall_seconds is not None else None
        if self.stall_seconds is not None and self.stall_seconds <= 0:
            raise ValueError("stall_seconds must be positive or None")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-alignment")
        self._sessions = set()
        self._sessions_lock = threading.Lock()

    @property
    def timestamp_levels(self):
        return list(dict.fromkeys(level for backend in (self.backend, self.multilingual_backend)
                                  if backend is not None
                                  for level in getattr(backend, "timestamp_levels", ("word", "character"))))

    @property
    def language_timestamps(self):
        result = {}
        for backend in (self.multilingual_backend, self.backend):
            if backend is None:
                continue
            languages = getattr(backend, "languages", ())
            if getattr(backend, "language", None):
                languages = (backend.language,)
            for language in languages:
                result[language] = list(getattr(backend, "timestamp_levels", ("word", "character")))
        return result

    def start(self, text, sample_rate, *, callback=None, stop_on_completion=True, language=None,
              start_after=None, cancel_event=None):
        backend = self.backend
        language = language.lower().replace("_", "-") if language else None
        language = {"en": "english", "de": "german", "es": "spanish", "fr": "french",
                    "it": "italian", "pt": "portuguese", "pt-br": "portuguese",
                    "ru": "russian"}.get(language, language)
        supported = getattr(backend, "language", None)
        if supported is not None and language is not None and language != supported:
            backend = self.multilingual_backend
            if backend is None:
                raise ValueError(f"Alignment model supports {supported}, not {language}")
        languages = getattr(backend, "languages", ())
        if languages and language not in languages:
            raise ValueError(f"Alignment model does not support {language}")
        prepare = getattr(backend, "prepare_text", None)
        if callable(prepare):
            text, projection = prepare(text)
            if projection is not None and callback is not None:
                callback = projection(callback)
        normalized, targets = backend.encode(text)
        session = StreamingAlignmentSession(
            self, normalized, targets, sample_rate, callback, stop_on_completion,
            backend=backend, start_after=start_after, cancel_event=cancel_event,
        )
        with self._sessions_lock:
            self._sessions.add(session)
        session.future = self._executor.submit(session._run)
        return session

    def close(self):
        with self._sessions_lock:
            sessions = tuple(self._sessions)
        for session in sessions:
            session.finish(cancelled=True)
        self._executor.shutdown(wait=True)


class StreamingAlignmentSession:
    def __init__(self, owner, text, targets, sample_rate, callback, stop_on_completion,
                 *, backend=None, start_after=None, cancel_event=None):
        self.owner = owner
        self.backend = backend if backend is not None else owner.backend
        self._start_after = start_after
        self._cancel_event = cancel_event
        self.interval_seconds = max(owner.interval_seconds, getattr(self.backend, "interval_seconds", 0))
        self.sample_rate = int(sample_rate)
        self.callback = callback
        self.stop_on_completion = stop_on_completion
        self.tracker = CTCPrefixTracker(text, targets, self.backend.blank_id,
                                        separator_id=getattr(self.backend, "separator_id", None))
        self.stop_requested = threading.Event()
        self.done = threading.Event()
        self._condition = threading.Condition()
        self._audio = np.empty(0, dtype=np.float32)
        self._end = 0
        self._finished = False
        self._cancelled = False
        self._published_characters = 0
        self._published_words = 0
        self.characters = []
        self.words = []
        self.complete = False
        self.error = None
        self.inference_ms = 0.0
        self.inference_calls = 0
        self.analyzed_seconds = 0.0
        self.finish_reason = None
        self.future = None

    def feed(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        with self._condition:
            if self._finished or self.done.is_set():
                return
            self._end += audio.size
            limit = int(self.owner.window_seconds * self.sample_rate)
            self._audio = np.concatenate((self._audio, audio))[-limit:]
            self._condition.notify()

    def finish(self, *, cancelled=False):
        with self._condition:
            self._finished = True
            self._cancelled = bool(cancelled)
            self._condition.notify()

    def _emit(self, event):
        if self.callback is not None:
            try:
                self.callback(event)
            except Exception:
                logging.exception("TTS alignment callback failed")

    def _publish_measured(self, final=False):
        watermark = self.analyzed_seconds - self.backend.right_context_seconds
        stable_chars = [
            c for c in self.characters
            if final or self.complete or c.end_time < watermark
        ]
        stable_words = [
            w for w in self.words
            if final or self.complete or w.end_time < watermark
        ]
        new_chars = stable_chars[self._published_characters:]
        new_words = stable_words[self._published_words:]
        self._published_characters += len(new_chars)
        self._published_words += len(new_words)
        if "character" not in getattr(self.backend, "timestamp_levels", ("word", "character")):
            new_chars = []
        if new_chars or new_words:
            self._emit({
                "type": "alignment", "words": [asdict(w) for w in new_words],
                "characters": [asdict(c) for c in new_chars],
                "analyzed_seconds": self.analyzed_seconds, "complete": self.complete,
            })
        return stable_chars

    def _run(self):
        processed = 0
        progress_index, progress_time = -1, 0.0
        step = self.backend.frame_seconds
        text, targets = self.tracker.text, self.tracker.targets
        try:
            if self._start_after is not None:
                while not self._start_after.wait(0.01):
                    if (self._cancelled or (self._finished and self._end == 0)
                            or (self._cancel_event is not None and self._cancel_event.is_set())):
                        self._cancelled = self._cancelled or bool(self._cancel_event is not None and self._cancel_event.is_set())
                        return
            while True:
                with self._condition:
                    hop = int(self.interval_seconds * self.sample_rate)
                    self._condition.wait_for(
                        lambda: self._finished or self._end - processed >= hop
                    )
                    if self._cancelled:
                        return
                    available, final = self._end, self._finished
                    end = available if final else available - available % hop
                    if end == processed:
                        # No new PCM is needed to release already measured timing
                        # held for right context at the end of this fragment.
                        if final:
                            self._publish_measured(final=True)
                        return
                    audio = self._audio[:len(self._audio) - (available - end)].copy()
                    buffer_start = end - audio.size
                # Re-evaluate overlapping audio: a non-streaming CTC model can
                # revise its language/characters once more context is available.
                offset, start, anchor = 0, 0, 0.0
                context_start = end / self.sample_rate - 3.2
                for match, word in zip(self.tracker.words, self.words):
                    if word.start_time > context_start:
                        break
                    offset = match.start()
                    anchor = word.start_time
                    start = max(0, int((word.start_time - 0.08) * self.sample_rate))
                if start < buffer_start:
                    raise RuntimeError("Alignment worker fell behind its audio window")
                stride = max(1, round(step * self.sample_rate))
                start -= start % stride
                audio = audio[max(0, start - buffer_start):]
                started = time.perf_counter()
                verified_end_time = None
                if hasattr(self.backend, "align"):
                    characters, words, complete = self.backend.align(
                        audio, self.sample_rate, text[offset:],
                        offset_seconds=start / self.sample_rate, anchor_seconds=anchor,
                    )
                else:
                    emissions = self.backend.emissions(audio, self.sample_rate)
                    positions = start / self.sample_rate + np.arange(len(emissions)) * step
                    keep = positions >= anchor - 1e-6
                    # The synthesis transcript is already known. Align it directly;
                    # a separately decoded spelling must not veto its endpoint.
                    characters = display_ctc_characters(
                        emissions[keep], targets[offset:], text[offset:],
                        positions[keep], self.backend.blank_id, step)
                    complete = bool(characters) and len(characters) == len(text) - offset
                    if complete:
                        verified_end_time = characters[-1].end_time
                self.inference_ms += (time.perf_counter() - started) * 1000
                self.inference_calls += 1
                with self._condition:
                    if self._cancelled:
                        return
                characters = [CharacterAlignment(c.character, c.start_time, c.end_time,
                                                 c.index + offset, c.score) for c in characters]
                prefix = [c for c in self.characters if c.index < offset]
                self.characters = prefix + characters
                by_index = {c.index: c for c in self.characters}
                self.words = []
                for match in self.tracker.words:
                    covered = [by_index.get(i) for i in range(match.start(), match.end())]
                    if all(c is not None for c in covered):
                        self.words.append(WordAlignment(match.group(), covered[0].start_time,
                                                        covered[-1].end_time, min(c.score for c in covered)))
                self.complete = complete
                self.analyzed_seconds = end / self.sample_rate
                stable_chars = self._publish_measured(final)
                if complete and self.characters:
                    end_time = (verified_end_time if verified_end_time is not None
                                else self.characters[-1].end_time)
                    self.finish_reason = "verified_text_end"
                    if self.stop_on_completion:
                        self.stop_requested.set()
                    self._emit({
                        "type": "alignment_complete", "end_time": end_time,
                        "trim_end_time": end_time + self.owner.tail_seconds,
                        "stop_requested": self.stop_on_completion,
                        "analyzed_seconds": self.analyzed_seconds,
                        "finish_reason": self.finish_reason,
                    })
                    return
                if stable_chars and stable_chars[-1].index > progress_index:
                    progress_index = stable_chars[-1].index
                    progress_time = stable_chars[-1].end_time
                if (not final and self.stop_on_completion and self.owner.stall_seconds is not None
                        and self.analyzed_seconds - progress_time >= self.owner.stall_seconds):
                    # An unresolved prefix is an explicit stall, never a claim
                    # that all requested words were spoken.
                    self.finish_reason = "alignment_stalled"
                    self.stop_requested.set()
                    self._emit({
                        "type": "alignment_stalled", "finish_reason": self.finish_reason,
                        "characters_confirmed": progress_index + 1,
                        "analyzed_seconds": self.analyzed_seconds,
                    })
                    return
                processed = end
                if final:
                    return
        except Exception as exc:
            self.error = str(exc)
            self._emit({"type": "alignment_error", "message": self.error})
            logging.warning("Streaming TTS alignment unavailable: %s", exc)
        finally:
            self._emit({
                "type": "alignment_done", "complete": self.complete,
                "cancelled": self._cancelled, "finish_reason": self.finish_reason,
                "inference_ms": self.inference_ms, "inference_calls": self.inference_calls,
                "audio_seconds": self._end / self.sample_rate, "error": self.error,
            })
            self.done.set()
            with self.owner._sessions_lock:
                self.owner._sessions.discard(self)
