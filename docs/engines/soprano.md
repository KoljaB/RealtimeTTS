# Soprano Engine

`SopranoEngine` wraps SopranoTTS. It is a single-voice English engine with
streaming and full-waveform APIs, but it does not currently provide voice
cloning.

## Install

```bash
pip install "realtimetts[soprano]"
```

For CUDA, install a compatible torch stack for your backend. The upstream
package supports `auto`, `transformers`, and `lmdeploy` backends.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, SopranoEngine


if __name__ == "__main__":
    engine = SopranoEngine(backend="transformers", device="cuda")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Soprano.")
    stream.play()
    engine.shutdown()
```

## Source Notes

- `SopranoVoice` is a placeholder because the current upstream model is
  single-voice.
- Defaults include `backend="auto"`, `device="auto"`, `cache_size_mb=100`,
  `decoder_batch_size=1`, `chunk_size=1`, `top_p=0.95`,
  `temperature=0.0`, and `streaming=True`.
- When `streaming=True`, the wrapper queues `infer_stream(...)` chunks. When
  false, it queues full `infer(...)` output.
- Output is mono 16-bit PCM at 32000 Hz.

## Zaphod Dev-Log Notes

- Official package checked was `soprano-tts==0.2.0`, Python `>=3.10`,
  Apache-2.0.
- A working Transformers CUDA path ended with `torch==2.10.0+cu128`,
  `torchvision==0.25.0+cu128`, `lmdeploy==0.12.3`,
  `transformers==4.57.1`, and `triton-windows`.
- `transformers==4.53.3` was too old for Soprano's Qwen3 config.
  `transformers==4.57.0` matched but was yanked; `4.57.1` worked in the Zaphod
  env.
- LMDeploy was investigated but the RealtimeTTS/Zaphod default was kept on the
  Transformers backend unless LMDeploy is explicitly requested.

## Troubleshooting

- If `backend="auto"` selects an unwanted backend, pass `backend="transformers"`
  or another explicit backend.
- If CUDA import fails after a timed-out install, repair the torch stack and run
  `pip check` before benchmarking.
