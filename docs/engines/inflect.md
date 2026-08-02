# Inflect Engine

`InflectEngine` integrates the quality-focused
[Inflect-Micro-v2](https://huggingface.co/owensong/Inflect-Micro-v2) model as a
small, low-latency local English voice. It uses optimized PyTorch on CUDA and
the official ONNX export on CPU.

## Install

```bash
pip install "realtimetts[inflect]"
```

The `inflect` extra installs both backends for automatic selection. Smaller
backend-specific installs are available when the deployment target is known;
select their backend explicitly in application code:

```bash
# CPU-only ONNX backend
pip install "realtimetts[inflect-onnx]"

# PyTorch deployment; does not install ONNX Runtime
pip install "realtimetts[inflect-pytorch]"
```

The first engine construction downloads a pinned model snapshot from Hugging
Face. Pass `cache_dir` to choose the cache location or `model_dir` to use an
already downloaded snapshot.

The Inflect ONNX engine path does not import or use PyTorch. A normal
RealtimeTTS installation can nevertheless install PyTorch transitively because
its standard `stream2sentence` dependency includes Stanza. Deployments that
must exclude Torch need to control those base dependencies separately.

## Minimal Use

```python
from RealtimeTTS import InflectEngine, TextToAudioStream


if __name__ == "__main__":
    engine = InflectEngine()
    stream = TextToAudioStream(engine)
    stream.feed("A small local voice can still respond very quickly.")
    stream.play()
    engine.shutdown()
```

`backend="auto"` is the default. With the full `inflect` extra, it selects
PyTorch with `cuda:0` when the complete PyTorch runtime is available and ONNX
with the CPU provider otherwise. Pip does not record which extra selected a
dependency, so explicitly pass `backend="pytorch"` or `backend="onnx"` when
using a backend-specific extra or a shared environment with unrelated Torch
packages.

Explicit CUDA and CPU configurations are:

```python
# Recommended GPU path
cuda_engine = InflectEngine(backend="pytorch", device="cuda:0")

# Recommended CPU path; tune the thread count for the target machine
cpu_engine = InflectEngine(backend="onnx", device="cpu", cpu_threads=8)
```

## Voice Controls

Inflect-Micro-v2 contains one fixed synthetic English male voice. It does not
clone voices or expose pitch control. The supported synthesis parameters are:

- `speed`: `0.5` through `2.0`; default `1.0`.
- `variation`: `0.0` through `1.0`; default `0.667`.
- `seed`: non-negative integer random seed; default `0`.

They can be passed to the constructor or changed later:

```python
engine.set_voice_parameters(speed=1.08, variation=0.55, seed=7)
```

## Latency

Inflect runs substantially faster than real time on a supported CUDA GPU, but
character count alone is not a reliable latency predictor: phoneme count,
predicted audio duration, the first unseen tensor shape, GPU, PyTorch build, and
operating system all affect a call.

On an RTX 2080 SUPER with PyTorch 2.6 CUDA, distinct 100- and 150-character
development samples generally completed `InflectEngine.synthesize()` in about
0.09 to 0.17 seconds after initial shape allocation. One first 100-character
shape took about 0.56 seconds despite constructor warmup, and separate cold
runs have varied roughly from 0.4 to 0.7 seconds. A 48-character first sentence
reached RealtimeTTS's audio callback in about 0.15 seconds after both
`InflectEngine` and `TextToAudioStream` had been constructed. Treat these as
orientation measurements, not a service-level guarantee, and benchmark the
actual deployment texts and hardware.

The engine warms the model during construction by default. It also asks
`TextToAudioStream` to initialize its standard sentence tokenizer during stream
construction so that one-time tokenizer setup is not charged to the first
`play()` call. Other engines retain their existing lazy tokenizer behavior. Set
`warmup=False` only when delayed first-speech latency is acceptable.

The upstream API returns a complete waveform, not native audio chunks. In
RealtimeTTS, sentence fragments provide effective streaming: keep the first
fragment short when time to first audio matters. Stopping during inference
discards the completed waveform; it cannot interrupt the model mid-call.

## Implementation Notes

- Output is mono 16-bit PCM at 24 kHz.
- The model and eSpeak frontend remain loaded between fragments.
- The engine caches the eSpeak backend that otherwise adds substantial fixed
  preprocessing latency, while preserving the upstream phoneme output.
- Calls across Inflect engine instances are serialized for deterministic
  seeded PyTorch synthesis.
- PyTorch CPU and CUDA random-generator states are restored after each call, so
  sequential host-model code is not left on Inflect's seed.
- The tested PyTorch and ONNX Hugging Face revisions are fixed by the engine;
  downloaded model files and executable helpers are checksum-verified by
  default.
- Set `local_files_only=True` for offline startup after the snapshot is cached.
- Set `verify_files=False` only when intentionally using modified files in the
  pinned model directory.

The Inflect code and model weights are Apache-2.0. The optional install also
resolves separate dependencies, including GPLv3+ `phonemizer`, GPLv2+
`Unidecode`, LGPL `num2words`, and eSpeak-ng through `espeakng-loader`; they are
not bundled into the RealtimeTTS wheel. Review their complete license and
redistribution obligations for your product.
