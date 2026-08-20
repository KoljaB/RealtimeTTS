"""Run a real QwenEngine installation smoke test and report its speed.

Example::

    python tests/qwen_test.py --play

By default this uses the repository's ``tests/zipvoice_reference1.wav`` in
x-vector mode, so no transcript is needed.  Pass both ``--ref-audio`` and the
recording's exact ``--ref-text`` to test full ICL cloning.  The test uses the
native Qwen backend through RealtimeTTS, writes the generated 24 kHz mono WAV,
and measures both time to the first audible queued audio and the engine's
real-time factor (RTF).  Use ``--play`` to send the generated audio to the
default speaker as well as saving it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

DEFAULT_TEXT = "This is a Qwen text to speech installation test."
DEFAULT_REFERENCE_AUDIO = Path(__file__).with_name("zipvoice_reference1.wav")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=DEFAULT_REFERENCE_AUDIO,
        help="Reference WAV used for voice cloning (default: tests/zipvoice_reference1.wav).",
    )
    parser.add_argument(
        "--ref-text",
        help="Exact transcript of the reference audio; omit it for x-vector cloning.",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text to synthesize (default: %(default)r).",
    )
    parser.add_argument(
        "--language",
        default="english",
        help="Qwen language for the generated speech (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qwen_test.wav"),
        help="Output WAV path (default: %(default)s).",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Also play the generated audio through the default speaker.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Qwen sampling seed (default: 42 for reproducible installation tests; use -1 for random output).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    reference_audio = args.ref_audio.expanduser().resolve()
    if not reference_audio.is_file():
        parser.error(f"reference audio does not exist: {reference_audio}")
    ref_text = args.ref_text.strip() if args.ref_text else None
    if not args.text.strip():
        parser.error("--text must not be empty")

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    engine = None
    first_audio_ns: list[int] = []
    started_ns = time.perf_counter_ns()

    def on_audio_stream_start() -> None:
        first_audio_ns.append(time.perf_counter_ns())

    try:
        from RealtimeTTS import QwenEngine, QwenVoice, TextToAudioStream

        voice = QwenVoice(
            name="installation-test",
            ref_audio=reference_audio,
            ref_text=ref_text,
            language=args.language,
        )
        print("Loading QwenEngine and preparing the reference voice...")
        engine = QwenEngine(voice=voice, seed=args.seed)
        stream = TextToAudioStream(
            engine,
            muted=not args.play,
            on_audio_stream_start=on_audio_stream_start,
        )
        print(f"Synthesizing to {output_path} ...")
        started_ns = time.perf_counter_ns()
        stream.feed(args.text).play(
            output_wavfile=str(output_path),
            minimum_sentence_length=1,
            minimum_first_fragment_length=1,
        )

        profile = dict(engine.last_synthesis_profile)
        if engine.last_error is not None:
            raise RuntimeError(f"Qwen synthesis failed: {engine.last_error}")
        if not output_path.is_file() or output_path.stat().st_size <= 44:
            raise RuntimeError(f"Qwen synthesis produced no WAV audio: {output_path}")

        audio_duration_s = float(profile.get("audio_duration_s", 0.0))
        total_ms = float(profile.get("total_ms", 0.0))
        if audio_duration_s <= 0.0:
            raise RuntimeError("Qwen synthesis produced zero seconds of audio")
        rtf = (total_ms / 1000.0) / audio_duration_s

        if first_audio_ns:
            first_audio_ms = (first_audio_ns[0] - started_ns) / 1_000_000
            first_audio = f"{first_audio_ms:.0f} ms"
        else:
            first_audio = "n/a"

        print(f"First audio latency: {first_audio}")
        print(f"First audible queued audio: {profile.get('first_queue_ms', 'n/a')} ms")
        print(f"Leading silence removed: {profile.get('leading_trimmed_ms', 0.0):.1f} ms")
        print(f"First queued audio duration: {profile.get('first_chunk_duration_ms', 'n/a')} ms")
        print(f"Minimum immediate-play margin: {profile.get('minimum_playout_margin_ms', 'n/a')} ms")
        print(f"Predicted immediate-play underruns: {profile.get('predicted_underruns', 'n/a')}")
        print(f"Native peak: {profile.get('native_peak', 'n/a')}")
        print(f"Native over-range samples: {profile.get('native_overrange_samples', 'n/a')}")
        print(f"Native maximum sample jump: {profile.get('native_max_sample_jump', 'n/a')}")
        print(f"Generated audio: {audio_duration_s:.2f} s")
        print(f"Synthesis time: {total_ms / 1000.0:.2f} s")
        print(f"RTF: {rtf:.3f} (< 1.0 is faster than realtime)")
        print(f"Saved: {output_path}")
        return 0
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
