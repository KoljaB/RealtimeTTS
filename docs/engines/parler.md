# Parler Engine

`ParlerEngine` wraps Parler TTS with a text voice prompt. It is a local neural
engine and is usually a GPU-oriented experiment rather than the lightest first
engine.

## Install

Install RealtimeTTS with the Parler extra:

```bash
pip install "realtimetts[parler]"
```

Install the upstream Parler package in the same environment. It is distributed
from its source repository rather than as a normal PyPI dependency that
RealtimeTTS can safely declare for upload:

```bash
pip install git+https://github.com/huggingface/parler-tts.git
```

Install a PyTorch build that matches your device before using CUDA.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, ParlerEngine


if __name__ == "__main__":
    engine = ParlerEngine(
        model_name="parler-tts/parler-tts-mini-v1",
        device="cuda:0",
    )
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Parler.")
    stream.play()
```

## Source Notes

- The module imports `parler_tts` at top level, so missing Parler dependencies
  can prevent importing the engine module.
- Defaults are `model_name="parler-tts/parler-tts-mini-v1"`, `device="cuda:0"`,
  and `torch_dtype=torch.bfloat16`.
- `voice_prompt` is a natural-language description of the desired voice.
- The source uses `ParlerTTSStreamer` with `buffer_duration_s` and
  `play_steps_in_s`.
- Output is mono float32 at 44100 Hz.

## Troubleshooting

- If import fails, install the Parler package in the same environment as
  RealtimeTTS.
- If CUDA fails, test a tiny PyTorch CUDA script first and consider using
  `device="cpu"` only for non-realtime experiments.
