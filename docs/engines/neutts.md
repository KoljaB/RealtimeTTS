# NeuTTS Engine

`NeuTTSEngine` wraps NeuTTS, an on-device voice-cloning TTS stack. It can load a
reference voice from an explicit `NeuTTSVoice` or from a voices directory.

## Status

- Local neural voice-cloning engine.
- `realtimetts[neutts]` installs the base NeuTTS package.
- `realtimetts[neutts-gguf]` asks pip for NeuTTS optional `llama` and `onnx`
  extras.
- Optional GGUF/ONNX paths require NeuTTS extras and model assets.
- Low-latency streaming in the Zaphod notes required a GGUF backbone; the torch
  safetensors backbone returned audio only after full generation.

## Install

Install RealtimeTTS with the NeuTTS extra:

```bash
pip install "realtimetts[neutts]"
```

For GGUF streaming and ONNX codec support:

```bash
pip install "realtimetts[neutts-gguf]"
```

For CUDA torch in the Zaphod env, the repaired package set used:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 ^
  torch==2.11.0+cu128 torchaudio==2.11.0+cu128
```

For the fast GGUF path, the dev-log used a CUDA `llama-cpp-python` wheel from
the cu124 wheel index, then pinned NumPy back to `2.2.6` after the wheel
upgraded it:

```bash
pip install --force-reinstall --no-cache-dir --only-binary llama-cpp-python ^
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 ^
  llama-cpp-python
pip install "numpy==2.2.6"
```

If you use a local NeuTTS checkout instead of an installed package, make sure it
is importable in the active Python environment. The wrapper currently probes a
small number of local checkout paths, but an installed package or activated
editable checkout is clearer.

## Minimal Example

```python
from RealtimeTTS import TextToAudioStream, NeuTTSEngine, NeuTTSVoice


if __name__ == "__main__":
    voice = NeuTTSVoice(
        name="demo",
        ref_audio_path="reference.wav",
        ref_text="Exact transcript of the reference audio.",
    )

    engine = NeuTTSEngine(device="cuda", voice=voice)
    stream = TextToAudioStream(engine)
    stream.feed("Hello from NeuTTS.")
    stream.play()
    engine.shutdown()
```

## Voices Directory

You can also point `voices_dir` to a folder containing `.wav` files and matching
`.txt` transcript files:

```text
voices/
  demo.wav
  demo.txt
```

```python
engine = NeuTTSEngine(
    device="cuda",
    voices_dir="voices",
    default_voice="demo",
)
```

Each transcript must be the exact text spoken in the reference audio.

## Constructor Inputs

Important setup parameters:

| Parameter | Meaning |
| --- | --- |
| `model` | Friendly model name; source default is `neutts-nano`. |
| `backbone_repo` | Hugging Face repo, local GGUF file, or compatible backbone path. |
| `codec_repo` | Codec repo or path; source default is `neuphonic/neucodec`. |
| `device` | Default device for backbone and codec. |
| `backbone_device`, `codec_device` | Override devices separately. |
| `language` | Optional language override. |
| `voice` | Optional `NeuTTSVoice`. |
| `voices_dir` | Optional directory of `.wav` plus `.txt` transcript pairs. |
| `default_voice` | Name to select from `voices_dir`. |
| `streaming` and streaming parameters | Tune chunking and overlap behavior. |

## Zaphod Dev-Log Notes

- Initial `pip install neutts` installed CPU torch; CUDA
  `torch==2.11.0+cu128` and `torchaudio==2.11.0+cu128` were installed
  afterward and `pip check` passed.
- `punkt_tab` was missing in the NeuTTS venv; copying local tokenizer data fixed
  the benchmark startup. Offline startup could still print a non-fatal NLTK
  warning.
- Upstream `infer_stream` raised `NotImplementedError` for the torch backbone,
  so torch backbone TTFA was effectively full generation time.
- The Q4 GGUF path used `neuphonic/neutts-nano-q4-gguf`, `backbone_device="gpu"`
  exactly, `codec_device="cuda"`, and `streaming=True`.
- Verify the CUDA llama.cpp wheel with `llama_cpp.llama_print_system_info()` and
  confirm it prints CUDA and sees the GPU.
- The engine resolves cached Q4/Q8 GGUF files before repeated/offline runs when
  they are already in the Hugging Face cache.
- Q8 GGUF was left as a quality fallback if Q4 quality is not acceptable.

## Example Script

See `tests/neutts_test.py` for the current manual demo. It uses a local
`voices_dir` path that should be replaced with your own folder before running.

## Troubleshooting

- `NeuTTS is not installed`: install `neutts` or activate an editable NeuTTS
  checkout.
- Missing reference WAV: check `NeuTTSVoice.ref_audio_path` or the files in
  `voices_dir`.
- Missing transcript: every voice directory WAV needs a matching `.txt` file.
- Slow torch synthesis: use a GGUF backbone for actual streaming, not only
  `streaming=True` on the torch backbone.
- If GGUF streaming is still CPU-bound, check that `llama-cpp-python` is a CUDA
  wheel and that `backbone_device` is `gpu`.
