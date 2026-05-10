# OmniVoice Engine

`OmniVoiceEngine` wraps `k2-fsa/OmniVoice` for local multilingual voice cloning.
It requires a reference audio file and exact transcript.

## Install

```bash
pip install "realtimetts[omnivoice]"
```

Install a compatible PyTorch build for your CUDA or CPU environment. The source
import error suggests:

```bash
pip install torch torchaudio omnivoice
```

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, OmniVoiceEngine, OmniVoiceVoice


if __name__ == "__main__":
    voice = OmniVoiceVoice(
        name="demo",
        ref_audio="reference.wav",
        ref_text="Exact transcript of the reference audio.",
        language="en",
    )
    engine = OmniVoiceEngine(voice=voice)
    stream = TextToAudioStream(engine)
    stream.feed("Hello from OmniVoice.")
    stream.play()
```

## Source Notes

- Default model is `k2-fsa/OmniVoice`.
- Default `device_map` is `cuda:0` and default dtype is `torch.float16`.
- `num_steps_schedule` defaults to `[12, 32]`; the first sentence uses the
  first value and later sentences reuse later values.
- `preprocess_prompt` and `postprocess_output` are forwarded to generation.
- Output is mono 16-bit PCM at 24000 Hz.

## Zaphod Dev-Log Notes

- Isolated venv smoke tests found OmniVoice needed the `emotional_wavs`
  reference assets.
- Later speed tuning did not promote OmniVoice above the faster local paths in
  that environment.

## Troubleshooting

- `OmniVoiceEngine requires a voice configuration`: pass an `OmniVoiceVoice`
  with both `ref_audio` and `ref_text`.
- If CUDA memory is tight, reduce steps or test CPU only for correctness first.
