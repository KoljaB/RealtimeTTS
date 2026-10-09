"""CPU CTC emissions for OmniASR or a language-specific Wav2Vec2 ONNX export."""
from __future__ import annotations

from pathlib import Path
import json
import re
import unicodedata
import numpy as np


class OnnxCTCBackend:
    sample_rate = 16000
    frame_seconds = 0.02
    right_context_seconds = 0.16

    def __init__(self, model_dir, *, threads: int = 2, language=None):
        import onnxruntime as ort
        from scipy.signal import resample_poly

        root = Path(model_dir)
        self.language = language.lower() if language else None
        if self.language == "english":
            # Load during service preparation, never at first spoken text.
            from num2words import num2words
            self._number_to_words = num2words
        if (root / "vocab.json").is_file():
            if self.language is None:
                raise ValueError("A Wav2Vec2 alignment model requires alignment_language")
            vocabulary = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
            self.tokens = {token.lower(): index for token, index in vocabulary.items()}
            self.tokens[" "] = vocabulary["|"]
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            self.blank_id = int(config["pad_token_id"])
            self.normalization_epsilon = 1e-7
            model_file = root / "model_quantized.onnx"
        else:
            self.tokens = {}
            for line in (root / "tokens.txt").read_text(encoding="utf-8").splitlines():
                token, index = line.rsplit(" ", 1)
                self.tokens[token] = int(index)
            self.blank_id = 0
            self.normalization_epsilon = 1e-5
            model_file = root / "model.int8.onnx"
        self.separator_id = self.tokens.get(" ")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(model_file), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self._resample = resample_poly
        # Warm the acoustic model at construction, never on the first audio chunk.
        self.emissions(np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate)


    def prepare_text(self, text):
        """Align English numerals as spoken words, then restore their source spans."""
        if self.language != "english" or not any(c.isdigit() for c in text):
            return text, None
        parts, groups, words, offset, source_offset = [], [], [], 0, 0
        for match in re.finditer(r"\d{1,2}:\d{2}|[\w']+", unicodedata.normalize("NFKC", text).lower()):
            source = match.group()
            numeric = any(c.isdigit() for c in source)
            if re.fullmatch(r"\d{1,2}:00", source) and int(source[:-3]) < 24:
                spoken = self._number_to_words(int(source[:-3]), lang="en") + " hundred"
            else:
                spoken = re.sub(r"\d+", lambda m: " " + self._number_to_words(int(m.group()), lang="en") + " ", source)
            spoken = " ".join("".join(c if c.isalnum() or c == "'" else " " for c in spoken).split())
            indexes = [offset + i for i, c in enumerate(spoken) if not c.isspace()]
            if numeric:
                groups.append((source, source_offset, indexes))
            else:
                groups.extend((c, source_offset + i, [offset + i]) for i, c in enumerate(spoken))
            words.append((source, source_offset, indexes))
            parts.append(spoken)
            offset += len(spoken) + 1
            source_offset += len(source) + 1
        return " ".join(parts), lambda callback: _source_timestamps(callback, groups, words)

    def encode(self, text: str):
        normalized = " ".join(
            "".join(c if c.isalnum() or c == "'" else " "
                    for c in unicodedata.normalize("NFKC", text).lower()).split()
        )
        if not normalized:
            raise ValueError("No pronounceable text to align")
        missing = {c for c in normalized if c not in self.tokens}
        if missing:
            raise ValueError("Alignment vocabulary does not cover the input text")
        return normalized, [self.tokens[c] for c in normalized]

    def emissions(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import math

        x = np.asarray(audio, dtype=np.float32)
        if sample_rate != self.sample_rate:
            gcd = math.gcd(sample_rate, self.sample_rate)
            x = self._resample(x, self.sample_rate // gcd, sample_rate // gcd).astype(np.float32)
        if x.size < 400:
            return np.empty((0, len(self.tokens)), dtype=np.float32)
        x = (x - x.mean()) / np.sqrt(x.var() + self.normalization_epsilon)
        logits = self.session.run(None, {self.input_name: x[None, :]})[0][0]
        logits = logits.astype(np.float32, copy=False)
        maximum = logits.max(axis=1, keepdims=True)
        shifted = logits - maximum
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

def _source_timestamps(callback, character_groups, word_groups):
    """One measured interval per numeral; never invent per-digit timing."""
    measured, cursors = {}, {"characters": 0, "words": 0}
    def projected(event):
        if event.get("type") != "alignment":
            return callback(event)
        measured.update((c["index"], c) for c in event.get("characters", ()))
        result = dict(event)
        for key, groups, label in (("characters", character_groups, "character"), ("words", word_groups, "word")):
            result[key] = []
            while cursors[key] < len(groups):
                value, index, positions = groups[cursors[key]]
                if not positions or any(p not in measured for p in positions):
                    break
                spans = [measured[p] for p in positions]
                item = {label: value, "start_time": min(c["start_time"] for c in spans),
                        "end_time": max(c["end_time"] for c in spans),
                        "score": min(c["score"] for c in spans)}
                if key == "characters":
                    item["index"] = index
                result[key].append(item)
                cursors[key] += 1
        if result["characters"] or result["words"]:
            callback(result)
    return projected
