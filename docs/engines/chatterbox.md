# Chatterbox Engine

`ChatterboxEngine` wraps Chatterbox Turbo. It supports a prompt-audio voice
workflow, but the checked public API returns a full waveform rather than a
native chunk iterator.

## Install

```bash
pip install "realtimetts[chatterbox]"
```

The setup extra installs `chatterbox-tts`. For CUDA, install a PyTorch build
compatible with Chatterbox and your platform.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, ChatterboxEngine, ChatterboxVoice


if __name__ == "__main__":
    voice = ChatterboxVoice(audio_prompt_path="reference.wav")
    engine = ChatterboxEngine(device="cuda", voice=voice)
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Chatterbox.")
    stream.play()
    engine.shutdown()
```

## Source Notes

- `ChatterboxVoice` accepts `audio_prompt_path`, `prompt_wav_path`, or
  `ref_audio`.
- Constructor defaults include `device="cuda"`, `repetition_penalty=1.2`,
  `top_p=0.95`, `temperature=0.8`, `top_k=1000`, and silence trimming enabled.
- `model_path` can point to a local model; otherwise the source calls
  `ChatterboxTurboTTS.from_pretrained(device=device)`.
- Reference conditioning is prepared with `prepare_conditionals(...)` and cached
  by path.
- Output is mono 16-bit PCM at the model sample rate.

## Zaphod Dev-Log Notes

- The candidate triage found Python `>=3.10` with Python 3.11 recommended by
  upstream, and package pins around `torch==2.6.0` / `torchaudio==2.6.0` for
  Python versions below 3.14.
- The Zaphod env eventually used `torch==2.6.0+cu124` with CUDA available.
- `pip install chatterbox-tts` initially hit network/timeouts and a stale Python
  file lock; stopping stale Python processes released the lock.
- Hugging Face download commands needed repeated `--include` flags for each
  pattern; one `--include` followed by many globs fetched only small files.
- Chatterbox prompt audio needed to be longer than 5 seconds in that pass.
- `HF_HOME=D:\hf_cache` emitted a warning when the hub directory was not
  writable; use a writable cache or local model path.
- Abort behavior is discard-after-generation, not true mid-inference
  cancellation.

## Troubleshooting

- If the prompt WAV is too short or low quality, prepare a longer reference.
- If low latency is the priority, remember that this wrapper waits for the full
  waveform before queueing audio.
