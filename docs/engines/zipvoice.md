# ZipVoice Engine

`ZipVoiceEngine` wraps a local ZipVoice checkout. It is a local neural
voice-cloning engine: each voice is defined by a prompt WAV file and the exact
transcript of that prompt.

## Status

- Local neural voice-cloning engine.
- `realtimetts[zipvoice]` installs the Python dependencies RealtimeTTS can
  declare, but the engine still needs a ZipVoice source checkout.
- Requires a ZipVoice source checkout importable through `zipvoice_root`.
- Downloads model files from Hugging Face by default unless local paths are
  provided.
- A Docker server example exists under `docker/zipvoice/`.
- Zaphod's accepted fast path used `model_name="zipvoice_distill"` with at
  least 3 diffusion steps. 1-step and 2-step probes are marked historical only.

## Direct Python Setup

Install RealtimeTTS with the ZipVoice extra. The exact ZipVoice checkout install
command should still follow the upstream ZipVoice repository you are using.

```bash
pip install "realtimetts[zipvoice]"
```

Then clone and install ZipVoice separately. The RealtimeTTS wrapper expects the
root directory of that checkout:

```python
from RealtimeTTS import TextToAudioStream, ZipVoiceEngine, ZipVoiceVoice


if __name__ == "__main__":
    voice = ZipVoiceVoice(
        prompt_wav_path="reference.wav",
        prompt_text="Exact transcript of the reference audio.",
    )

    engine = ZipVoiceEngine(
        zipvoice_root="D:/path/to/ZipVoice",
        voice=voice,
        model_name="zipvoice",
        device="cuda",
    )

    stream = TextToAudioStream(engine)
    stream.feed("Hello from ZipVoice.")
    stream.play()
    engine.shutdown()
```

## Constructor Inputs

Important setup parameters:

| Parameter | Meaning |
| --- | --- |
| `zipvoice_root` | Root directory of the ZipVoice repository checkout. Required. |
| `voice` | `ZipVoiceVoice(prompt_wav_path, prompt_text)`. Required. |
| `model_name` | `zipvoice` or `zipvoice_distill`. |
| `checkpoint` | Optional local model checkpoint. If omitted, downloads from Hugging Face. |
| `model_config` | Optional local model config JSON. If omitted, downloads from Hugging Face. |
| `vocoder_path` | Optional local Vocos directory. If omitted, uses the pretrained Vocos model. |
| `token_file` | Optional local token file. If omitted, downloads from Hugging Face. |
| `tokenizer_type` | `emilia`, `libritts`, `espeak`, or `simple`. |
| `language` | Used with the `espeak` tokenizer. |
| `device` | `cuda`, `mps`, or `cpu`; source defaults to `cuda`. |

ZipVoice caches prepared prompt features under `zipvoice_voices/` so repeated
runs with the same reference prompt can start faster.

## Model Assets

When paths are not supplied, the source downloads from Hugging Face repo
`k2-fsa/ZipVoice`:

| Model | Files |
| --- | --- |
| `zipvoice` | `zipvoice/model.pt`, `zipvoice/tokens.txt`, `zipvoice/zipvoice_base.json` |
| `zipvoice_distill` | `zipvoice_distill/model.pt`, `zipvoice_distill/tokens.txt`, `zipvoice_distill/zipvoice_base.json` |

The default vocoder is `charactr/vocos-mel-24khz` unless `vocoder_path` is
provided.

## Zaphod Dev-Log Notes

- ZipVoice required `third_party\ZipVoice` in the local Zaphod layout.
- The distill files were downloaded from `k2-fsa/ZipVoice` into
  `D:\hf_cache\hub`.
- Native Windows `k2` CUDA wheels were not available for the checked
  `win_amd64` path, so k2 acceleration was blocked unless compiling from source
  or using WSL/Linux.
- Early slow results used the full model at 16 steps without k2. The current
  accepted speed preset in Zaphod was `zipvoice_distill`, `num_step=3`,
  `guidance_scale=3.0`, `speed=1.0`, and warmup across all mixed texts.
- Do not rerun 1-step or 2-step ZipVoice inference as quality references; those
  outputs were retired as low-quality speed probes.
- Prompt caches now include reference audio identity, transcript, target RMS,
  feature scale, sampling rate, and cache version. Legacy basename-only caches
  should be ignored.
- NLTK `punkt_tab` network errors were seen during startup, but configured local
  NLTK data/fallback paths let the real smokes and benchmarks complete.

## Docker Server

The Docker example is documented in `docker/zipvoice/README.md`.

```bash
docker build -t zipvoice-image -f docker/zipvoice/Dockerfile .
docker run --rm --name zipvoice-container -p 9086:9086 --gpus all zipvoice-image
```

The server exposes a streaming endpoint on port `9086` and uses prompt voices
configured by the Docker server script.

## Example Script

See `tests/zipvoice_test.py` for a manual demo that switches between two
`ZipVoiceVoice` prompts and writes WAV outputs.

## Troubleshooting

- `zipvoice_root is not a valid directory`: pass the root of the ZipVoice
  checkout, not the RealtimeTTS repository.
- Import errors from `zipvoice.*`: install the upstream ZipVoice dependencies
  into the active environment.
- Slow first run: model files and prompt features may be downloading or being
  cached.
- CPU mode can be much slower than CUDA for realtime playback.
- If quality is poor, verify that the prompt transcript is exact and use at
  least 3 distill steps or the full model rather than retired 1-step/2-step
  settings.
