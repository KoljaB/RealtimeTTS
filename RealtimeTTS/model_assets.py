"""Pinned optional model downloads, cached by huggingface_hub."""
from pathlib import Path


def breeze_model():
    from huggingface_hub import hf_hub_download
    print("Preparing Breeze Q8_0 model (downloaded once, then cached)...", flush=True)
    return Path(hf_hub_download(
        "HoppouAI/Breeze-TTS-2.cpp", "breeze-tts-2-q8_0.gguf",
        revision="81b22bad9f05b99970e30c5ee5e4bbc52fedf2f8",
    ))


def alignment_model(*, local_files_only=False):
    from huggingface_hub import snapshot_download
    print("Preparing multilingual highlighting model (downloaded once, then cached)...", flush=True)
    return Path(snapshot_download(
        "csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12",
        revision="6abf1ece20cd2308bdb7d13cd78ec1c44fa4c094",
        allow_patterns=["model.int8.onnx", "tokens.txt"],
        local_files_only=local_files_only,
    ))


def prepare_studio_alignment(engine, *, local_files_only=False):
    """Display-only timestamps; no native decoder cutoff or audio buffering."""
    from RealtimeTTS.alignment.backends import CPUAlignmentModel
    from RealtimeTTS.alignment.onnx_ctc import OnnxCTCBackend
    cpu = getattr(engine, "device", None) == "cpu"
    existing = getattr(engine, "streaming_aligner", None)
    backend = getattr(existing, "backend", None)
    if backend is None:
        backend = OnnxCTCBackend(alignment_model(local_files_only=local_files_only), threads=1 if cpu else 2)
    affinity = ((getattr(engine, "cpu_affinity", 0) or 0)
                | (getattr(engine, "cpu_codec_affinity", 0) or 0)) if cpu else None
    engine.studio_aligner = CPUAlignmentModel(backend, cpu_affinity=affinity)
