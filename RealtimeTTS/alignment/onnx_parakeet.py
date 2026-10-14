"""CPU Parakeet TDT matching with native subword timestamps.

No forced decoding or text prompting: timestamps require exact recognized
words. Later matching words remain useful after a mismatch. Word times inherit
the model's 80 ms token resolution; these are
not separately measured character timestamps.
"""
from __future__ import annotations

from pathlib import Path
import math
from difflib import SequenceMatcher
import re
import unicodedata
import numpy as np

from .base import CharacterAlignment, WordAlignment


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("\u2019", "'")
    return " ".join("".join(c if c.isalnum() or c == "'" else " " for c in text).split())


def match_tokens(text, tokens, timestamps, logprobs, *, frame_seconds=0.08,
                 offset_seconds=0.0, anchor_seconds=0.0):
    """Match complete recognized words while retaining native subword times."""
    heard_words, current = [], []

    def finish_word():
        if current:
            # Discard a whole word crossing the committed anchor. Keeping its
            # suffix would turn a partial acoustic word into a false full match.
            if current[0][1] >= anchor_seconds - 1e-6:
                heard_words.append(tuple(current))
            current.clear()

    for token, timestamp, logprob in zip(tokens, timestamps, logprobs):
        start = offset_seconds + float(timestamp)
        score = math.exp(float(logprob))
        # Normalize punctuation without stripping leading/trailing whitespace:
        # that whitespace distinguishes "cat" from "caterpillar" and "a b".
        token = unicodedata.normalize("NFKC", token).lower().replace("\u2019", "'")
        for character in token:
            if character.isalnum() or character == "'":
                current.append((character, start, start + frame_seconds, score))
            else:
                finish_word()
    finish_word()

    expected = list(re.finditer(r"\S+", text))
    heard_text = ["".join(c for c, *_ in word) for word in heard_words]
    matcher = SequenceMatcher(
        None, [word.group() for word in expected], heard_text, autojunk=False)
    characters, words = [], []
    for block in matcher.get_matching_blocks():
        for position in range(block.size):
            target = expected[block.a + position]
            measured = heard_words[block.b + position]
            if any(not math.isfinite(c[3]) or c[3] < 0.1 for c in measured):
                continue
            characters.extend(
                CharacterAlignment(c, start, end, target.start() + i, score)
                for i, (c, start, end, score) in enumerate(measured)
            )
            words.append(WordAlignment(
                target.group(), measured[0][1], measured[-1][2],
                min(c[3] for c in measured)))

    by_index = {c.index: c for c in characters}
    for index, character in enumerate(text):
        before, after = by_index.get(index - 1), by_index.get(index + 1)
        if character.isspace() and before is not None and after is not None:
            by_index[index] = CharacterAlignment(
                character, before.end_time, max(before.end_time, after.start_time),
                index, min(before.score, after.score))
    characters = sorted(by_index.values(), key=lambda c: c.index)
    complete = bool(expected) and len(words) == len(expected)
    return characters, words, complete


class OnnxParakeetBackend:
    languages = ("english", "german", "spanish", "french", "italian", "portuguese", "russian")
    language = None
    blank_id = -1
    separator_id = ord(" ")
    sample_rate = 16000
    frame_seconds = 0.08
    right_context_seconds = 0.16
    interval_seconds = 0.32
    timestamp_levels = ("word",)

    def __init__(self, model_dir, *, threads=2):
        import onnxruntime as ort
        from onnx_asr.models.nemo import NemoConformerTdt
        from scipy.signal import resample_poly

        root = Path(model_dir)
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        cpu = {"sess_options": options, "providers": ["CPUExecutionProvider"]}
        self._preprocessor = ort.InferenceSession(str(root / "nemo128.onnx"), **cpu)
        names = [item.name for item in self._preprocessor.get_inputs()]

        def preprocess(audio, lengths):
            return tuple(self._preprocessor.run(None, {names[0]: audio, names[1]: lengths}))

        files = {key: root / name for key, name in {
            "encoder": "encoder-model.int8.onnx",
            "decoder_joint": "decoder_joint-model.int8.onnx",
            "vocab": "vocab.txt", "config": "config.json",
        }.items()}
        self.model = NemoConformerTdt(files, lambda _: preprocess, cpu)
        self._resample = resample_poly
        # Load and warm once at server startup, never before an utterance's first PCM.
        self.recognize(np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate)

    def encode(self, text):
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError("No pronounceable text to align")
        return normalized, [ord(c) for c in normalized]

    def recognize(self, audio, sample_rate):
        x = np.asarray(audio, dtype=np.float32)
        if sample_rate != self.sample_rate:
            gcd = math.gcd(sample_rate, self.sample_rate)
            x = self._resample(x, self.sample_rate // gcd, sample_rate // gcd).astype(np.float32)
        return next(iter(self.model.recognize_batch(
            x[None, :], np.array([x.size], dtype=np.int64), need_logprobs=True)))

    def align(self, audio, sample_rate, text, *, offset_seconds=0.0, anchor_seconds=0.0):
        result = self.recognize(audio, sample_rate)
        return match_tokens(text, result.tokens, result.timestamps, result.logprobs,
                            frame_seconds=self.frame_seconds, offset_seconds=offset_seconds,
                            anchor_seconds=anchor_seconds)
