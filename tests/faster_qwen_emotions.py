"""Stable entry point used in the published emotional Qwen video.

GPU: pip install -e ".[qwen]"; python faster_qwen_emotions.py
CPU: pip install -e ".[qwen-cpu]"; python faster_qwen_emotions.py --device cpu
Server: python faster_qwen_emotions.py --server http://localhost:8080
Headless: add --no-play to save WAV files without a playback device.

The old name is intentional. Do not remove/rename this advertised entry point.
"""
from pathlib import Path

from RealtimeTTS.qwen_emotions import main


if __name__ == "__main__":
    raise SystemExit(main(default_reference_dir=Path(__file__).resolve().parent / "ears_emotional_speaker11"))
