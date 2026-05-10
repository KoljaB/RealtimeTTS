# Faster Qwen Engine

`FasterQwenEngine` wraps `faster-qwen3-tts` for local Qwen voice cloning. It is
a CUDA-oriented engine that benefits from cached speaker embeddings.

## Install

```bash
pip install "realtimetts[qwen]"
```

The source notes that the `instruct` parameter needs `faster-qwen3-tts` 0.2.5.
If that version is not available on PyPI in your environment, install the
upstream package from Git.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, FasterQwenEngine, FasterQwenVoice


if __name__ == "__main__":
    voice = FasterQwenVoice(
        name="demo",
        ref_audio="reference.wav",
        ref_text="Exact transcript of the reference audio.",
        language="English",
        speaker_pt="speaker.pt",
    )
    engine = FasterQwenEngine(device="cuda", voice=voice)
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Faster Qwen.")
    stream.play()
```

## Source Notes

- Default model is `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.
- A `FasterQwenVoice` requires either `speaker_pt` or both `ref_audio` and
  `ref_text`.
- The engine extracts and caches the speaker embedding, then primes the
  underlying model cache with a sentinel key.
- Defaults include `chunk_size=2`, `xvec_only=True`, and
  `attn_implementation="sdpa"`.
- Output is mono 16-bit PCM at 24000 Hz.

## Zaphod Dev-Log Notes

- Faster Qwen was run in isolated engine environments because the shared venv
  became unstable across heavy model stacks.
- One benchmark path used a cached Hugging Face snapshot and offline mode.
- Later Zaphod tuning recorded the local cached Qwen snapshot path and exposed
  Qwen tuning fields in CLI/help output.

## Troubleshooting

- If first synthesis is slow, speaker extraction and CUDA graph warmup may be
  happening.
- If `instruct` does not work, check the installed `faster-qwen3-tts` version.
