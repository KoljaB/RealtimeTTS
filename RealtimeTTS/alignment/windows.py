"""Select a bounded acoustic window only behind a supported transcript prefix."""
from __future__ import annotations

import re


def window_anchor(text, characters, sample_rate, frame_seconds, end_seconds,
                  *, lookback_seconds=3.2, minimum_probability=.35):
    measured = {c.index: c for c in characters}
    offset, start, anchor = 0, 0, 0.0
    for match in re.finditer(r"\S+", text):
        covered = [measured.get(i) for i in range(match.start(), match.end())]
        if not covered or any(c is None or c.score is None for c in covered):
            continue
        # A weak individual letter must not pin all later audio to time zero.
        # Keep uncertain timestamps, but anchor only at a supported whole word.
        if sum(c.score >= minimum_probability for c in covered) / len(covered) < .5:
            continue
        if covered[0].start_time > end_seconds - lookback_seconds:
            break
        offset = match.start()
        anchor = covered[0].start_time
        start = max(0, int((anchor - .08) * sample_rate))
    # No transcript prefix has been committed at offset zero. A previous
    # estimate of the first word's onset must not exclude its revised onset.
    if offset == 0:
        return 0, 0, 0.0
    stride = max(1, round(frame_seconds * sample_rate))
    return offset, start - start % stride, anchor
