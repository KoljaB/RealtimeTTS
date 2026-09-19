"""Shared phoneme alignment and bounded boundary refinement for prefix splicing."""
from __future__ import annotations
import math
import os
import re
import unicodedata
import numpy as np

PHONE_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"

def words(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*", text)


def ctc_spans(log_probs, tokens, blank, allow_trailing=False):
    """Viterbi forced alignment, including mandatory blanks for repeated phones."""
    scores = np.asarray(log_probs)
    if scores.ndim != 2 or not tokens or not np.isfinite(scores).all():
        raise ValueError("Invalid emissions or empty phoneme sequence")
    if blank in tokens or any(t < 0 or t >= scores.shape[1] for t in tokens):
        raise ValueError("Invalid phoneme token")
    labels = np.full(2 * len(tokens) + 1, blank, dtype=int)
    labels[1::2] = tokens
    if allow_trailing:
        # An absorbing, unrestricted suffix: future words must NOT be forced
        # into the known prefix's final phone/blank. Relative frame scores keep
        # the suffix from favoring a shorter path merely because log p < 0.
        scores = scores - scores.max(axis=1, keepdims=True)
        labels = np.r_[labels, blank]
    prev = np.full(len(labels), -np.inf)
    prev[0] = scores[0, blank]
    prev[1] = scores[0, tokens[0]]
    back = np.zeros((len(scores), len(labels)), dtype=np.int8)
    skip = np.zeros(len(labels), dtype=bool)
    skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    for frame in range(1, len(scores)):
        advance = np.r_[-np.inf, prev[:-1]]
        jump = np.r_[-np.inf, -np.inf, prev[:-2]]
        jump[~skip] = -np.inf
        if allow_trailing:
            jump[-1] = prev[-3]  # last phone -> unrestricted suffix
        choices = np.stack((prev, advance, jump))
        back[frame] = choices.argmax(axis=0)
        emissions = scores[frame, labels].copy()
        if allow_trailing:
            emissions[-1] = 0
        prev = choices.max(axis=0) + emissions
    terminals = 3 if allow_trailing else 2
    state = len(labels)-1-int(np.argmax(prev[-terminals:][::-1]))
    if not np.isfinite(prev[state]):
        raise ValueError("Audio too short to align every phoneme")
    path = np.empty(len(scores), dtype=int)
    for frame in range(len(scores) - 1, -1, -1):
        path[frame] = state
        state -= int(back[frame, state])
    result = []
    for index, token in enumerate(tokens):
        frames = np.flatnonzero(path == 2 * index + 1)
        if not len(frames):
            raise ValueError("Alignment omitted a phoneme")
        result.append((int(frames[0]), int(frames[-1] + 1),
                       float(np.exp(np.asarray(log_probs)[frames, token].mean()))))
    return result


class PhoneAligner:
    def __init__(self, language, cache_dir):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from transformers.utils import logging
        logging.disable_progress_bar()
        self.torch = torch
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        self.processor = Wav2Vec2Processor.from_pretrained(PHONE_MODEL, cache_dir=cache_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(PHONE_MODEL, cache_dir=cache_dir).cpu().eval()
        self.language = language
        self.processor.tokenizer.phonemizer_lang = language
        self.processor.tokenizer.init_backend(language)
        self._word_tokens = {}

    def align_prefix(self, audio, rate, transcript, boundary_count):
        """Align through the first two phones of the word AFTER the boundary.

        Incomplete audio must never be forced to explain the complete sentence.
        Callers still require confidence, right context, and stable snapshots.
        """
        return self.align(audio, rate, " ".join(words(transcript)[:boundary_count+1]),
                          prefix_boundary=boundary_count)

    def align(self, audio, rate, transcript, prefix_boundary=None):
        from scipy.signal import resample_poly
        tokenizer = self.processor.tokenizer
        word_list = words(transcript)
        tokens, owners = [], []
        for index, word in enumerate(word_list):
            # The phoneme tokenizer's generic __call__ is broken in transformers
            # 5.2 (unexpected return_offsets_mapping); its token API is supported.
            if word not in self._word_tokens:
                if len(self._word_tokens) >= 2048:
                    self._word_tokens.clear()
                self._word_tokens[word] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            ids = self._word_tokens[word]
            if prefix_boundary is not None and index == prefix_boundary:
                ids = ids[:2]
            if not ids or tokenizer.unk_token_id in ids or tokenizer.pad_token_id in ids:
                raise ValueError(f"Unrepresentable phonemes in {word!r}")
            tokens.extend(ids)
            owners.extend([index] * len(ids))
        divisor = math.gcd(rate, 16000)
        waveform = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0].log_softmax(-1).numpy()
        spans = ctc_spans(logits, tokens, tokenizer.pad_token_id, allow_trailing=prefix_boundary is not None)
        # Wav2Vec2 emissions are spaced by the convolution stride, not duration/T.
        step = math.prod(self.model.config.conv_stride) / 16000
        phones = [{"phone": tokenizer.convert_ids_to_tokens(token), "word_index": owner,
                   "start": start * step, "end": end * step, "score": score}
                  for token, owner, (start, end, score) in zip(tokens, owners, spans)]
        timings = []
        for index, word in enumerate(word_list):
            group = [p for p in phones if p["word_index"] == index]
            timings.append({"word": word, "start": group[0]["start"],
                            "end": group[-1]["end"]})
        return {"model": PHONE_MODEL, "device": "cpu", "frame_seconds": step,
                "words": timings, "phones": phones}


def boundary_center(audio, rate, left_end, right_start, overlap, search_ms=5):
    """Refine a CTC inter-word midpoint using a local low-RMS plateau.

    CTC blanks are not silence. Stay inside the estimated inter-word interval
    and report the actual energy rather than assuming a silent boundary.
    """
    lo, hi = round(left_end*rate), round(right_start*rate)
    half = overlap // 2
    if hi-lo < overlap:
        raise ValueError("No room for the requested boundary crossfade; reduce --crossfade-ms")
    if not math.isfinite(search_ms) or not 0 <= search_ms <= 5:
        raise ValueError("Boundary correction must be between 0 and 5 ms")
    nominal = (lo+hi)/2
    radius = rate*search_ms/1000
    centers = np.arange(max(lo+half, math.ceil(nominal-radius)),
                        min(hi-(overlap-half), math.floor(nominal+radius))+1)
    if not len(centers):
        centers = np.array([round(nominal)])
    # RMS over the entire overlap, not a single zero crossing.
    energy = np.r_[0.0, np.cumsum(audio.astype(np.float64)**2)]
    powers = (energy[centers+overlap-half]-energy[centers-half])/overlap
    rms = np.sqrt(np.maximum(powers, 0))
    # A low plateau comprises windows within 15% of the observed RMS range
    # above the minimum. Among these choose the closest to the nominal center.
    low = rms <= rms.min() + .15*(rms.max()-rms.min()) + 1e-12
    candidate_indices = np.flatnonzero(low)
    selected = candidate_indices[np.argmin(abs(centers[candidate_indices]-(lo+hi)/2))]
    center = int(centers[selected])
    return center, {"nominal_center_seconds": (lo+hi)/(2*rate),
                    "refined_center_seconds": center/rate,
                    "correction_ms": (center-nominal)/rate*1000,
                    "max_correction_ms": search_ms,
                    "overlap_rms": float(rms[selected]),
                    "minimum_window_rms": float(rms.min()),
                    "search_start_seconds": float(centers[0])/rate,
                    "search_end_seconds": float(centers[-1])/rate}
