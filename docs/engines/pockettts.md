# PocketTTS Engines

`PocketTTSEngine` wraps Kyutai Labs' Pocket TTS package. The source describes it
as a lightweight CPU-oriented English model with optional voice cloning.

## Install

Install RealtimeTTS with the PocketTTS extra:

```bash
pip install "realtimetts[pockettts]"
```

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, PocketTTSEngine


if __name__ == "__main__":
    engine = PocketTTSEngine(voice="alba")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Pocket TTS.")
    stream.play()
```

Voice cloning:

```python
from RealtimeTTS import PocketTTSVoice, PocketTTSEngine

voice = PocketTTSVoice(name="custom", audio_prompt_path="reference.wav")
engine = PocketTTSEngine(voice=voice)
```

## CPU and GPU

PocketTTS is CPU-oriented, and `PocketTTSEngine` runs on CPU by default:

```python
engine = PocketTTSEngine(voice="alba")
```

This is equivalent to:

```python
engine = PocketTTSEngine(voice="alba", device="cpu")
```

To try CUDA, install a CUDA-enabled PyTorch build first, then pass a CUDA device:

```python
engine = PocketTTSEngine(voice="alba", device="cuda")
```

You can also use a concrete device such as `device="cuda:0"`. If CUDA is not
available in the active PyTorch install, initialization fails from PyTorch; use
`device="cpu"` in that environment. Voice states are created after the model is
moved to the selected device, so built-in voices and cloned voices use the same
device.

## CUDA Fork Engine

`PocketTTSGpuEngine` is a separate engine for the CUDA-capable
`pocket-tts-gpu` fork. It is intentionally separate from `PocketTTSEngine`
because it depends on a different runtime package.

Install RealtimeTTS' shared GPU dependencies:

```bash
pip install "realtimetts[pockettts-gpu]"
```

Install a CUDA-enabled PyTorch build that matches your NVIDIA driver and CUDA
runtime. The command below is an example for CUDA 12.6; use the selector at
https://pytorch.org/get-started/locally/ if you need a different build:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then install the pinned PocketTTS GPU fork revision used by this release:

```bash
pip install "git+https://github.com/Deveraux-Parker/kutai100temp.git@6beddc19c480da9ced9733ba0bb2f199f6e22ab4#subdirectory=pocket-tts-gpu"
```

Minimal use:

```python
from RealtimeTTS import TextToAudioStream, PocketTTSGpuEngine

engine = PocketTTSGpuEngine(voice="alba", device="cuda")
stream = TextToAudioStream(engine)
stream.feed("Hello from PocketTTS on the GPU.")
stream.play()
```

Voice cloning:

```python
from RealtimeTTS import PocketTTSGpuEngine, PocketTTSGpuVoice

voice = PocketTTSGpuVoice(name="custom", audio_prompt_path="reference.wav")
engine = PocketTTSGpuEngine(voice=voice, device="cuda")
```

If you use the gated `kyutai/pocket-tts` voice-cloning weights, set `HF_HOME`
to a cache containing that repository. When `voice_cache_dir` is writable, the
GPU engine can create a local config that points at the cached weights and
tokenizer.

## Source Notes

- Built-in source voices are `alba`, `marius`, `javert`, `jean`, `fantine`,
  `cosette`, `eponine`, and `azelma`.
- `PocketTTSVoice(name, audio_prompt_path=None)` treats a built-in name without
  a prompt as a built-in voice.
- The model is loaded with `TTSModel.load_model()` and then moved to the
  selected Torch device.
- Voice states are cached by voice name and prompt path.
- Output is mono 16-bit PCM at the model sample rate, falling back to 24000 Hz.
- `PocketTTSGpuEngine` defaults to `variant="b6369a24"` and `device="cuda"`.
- `PocketTTSGpuEngine` supports `teacher_forcing`, `frames_after_eos`,
  built-in voices, prompt WAVs, and cached prompt states.

## Runtime Notes

`PocketTTSEngine` keeps Pocket model work on a persistent synthesis worker and
patches Pocket's short-text streaming path to decode serially. This avoids a
Windows/Torch CPU memory-retention issue observed when repeated short
generations create fresh synthesis and decoder threads. The public
`TextToAudioStream` API remains unchanged; the worker is an internal engine
detail and is stopped by `shutdown()`.

## Troubleshooting

- `pocket-tts is not installed`: install `pocket-tts` in the active environment.
- `PocketTTS GPU dependencies are missing`: install the pinned CUDA fork with
  `pip install "git+https://github.com/Deveraux-Parker/kutai100temp.git@6beddc19c480da9ced9733ba0bb2f199f6e22ab4#subdirectory=pocket-tts-gpu"`.
- `torch.cuda.is_available()` is `False`: install a CUDA-enabled PyTorch build
  for your system, then rerun the CUDA availability check above.
- Unknown voice errors mean the name is not built in and no prompt WAV was
  supplied.
