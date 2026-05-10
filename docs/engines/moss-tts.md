# MOSS-TTS Engine

`MossTTSEngine` wraps MOSS-TTS-Nano ONNX and torch runtimes. The source keeps
the Realtime variant behind a clear error until a bounded local smoke exists.

## Install

Install RealtimeTTS with the MOSS extra. The extra includes the shared
PyPI-resolvable runtime dependencies RealtimeTTS can declare, but
MOSS-TTS-Nano itself, local model/runtime assets, and CUDA choices still need
deliberate setup.

```bash
pip install "realtimetts[moss]"
```

If you keep a local MOSS-TTS-Nano checkout, installing it editable remains a
good development workflow:

```bash
pip install -e third_party/MOSS-TTS-Nano
```

Choose one runtime path:

- ONNX: install the MOSS ONNX runtime dependencies and `onnxruntime` or
  `onnxruntime-gpu`.
- Torch: install the MOSS torch runtime dependencies and a compatible PyTorch
  and torchaudio build.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, MossTTSEngine, MossTTSVoice


if __name__ == "__main__":
    voice = MossTTSVoice(
        name="demo",
        prompt_audio_path="reference.wav",
        prompt_text="Exact transcript of the reference audio.",
    )
    engine = MossTTSEngine(voice=voice, backend="onnx", execution_provider="cpu")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from MOSS TTS.")
    stream.play()
    engine.shutdown()
```

## Source Notes

- Supported source variant is `variant="nano"` with `backend="onnx"` or
  `backend="torch"`.
- `variant="realtime"` or `backend="realtime"` currently raises an import error.
- Defaults target `OpenMOSS-Team/MOSS-TTS-Nano` and
  `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`.
- Default audio is 48000 Hz stereo; `get_stream_info()` returns channel count
  from the loaded runtime metadata.
- ONNX CUDA on Windows checks `MOSS_CUDA_DLL_DIRS`, `ORT_CUDA_DLL_DIRS`,
  `CUDA_PATH`, `CUDA_HOME`, and `CUDNN_PATH`, then tries common NVIDIA install
  directories and `onnxruntime.preload_dlls`.
- The wrapper writes temporary runtime output under `test_outputs/moss_tts_internal`
  by default.

## Zaphod Dev-Log Notes

- MOSS-TTS-Nano package metadata observed Python `>=3.10`,
  `torch==2.7.0`, `torchaudio==2.7.0`, `transformers==4.57.1`, and
  `onnxruntime>=1.20.0`.
- Working Nano venv used `torch==2.7.0+cu128`, `torchaudio==2.7.0+cu128`, and
  `onnxruntime-gpu`.
- `soundfile` was needed because the first Nano smoke had no working
  `torchaudio.load()` backend for the reference WAV.
- NLTK `punkt` and `punkt_tab` were installed locally for sentence splitting.
- Torch Nano rejects `prompt_text` in `voice_clone`; the wrapper records
  transcript data for reproducibility but does not pass it to that torch path.
- ONNX CPU, ONNX CUDA, and Torch CUDA smokes all produced valid files in the
  dev-log, but ONNX CUDA had instability and Torch CUDA had slower-than-realtime
  RTF despite good TTFA.
- The MOSS-TTS-Realtime probe used `torch==2.9.1+cu128`,
  `torchaudio==2.9.1+cu128`, and `torchcodec==0.8.1`, but full smoke attempts
  timed out or hit torchcodec/FFmpeg DLL issues.

## Troubleshooting

- For ONNX CUDA on Windows, set `CUDNN_PATH` or the explicit DLL directory env
  vars if CUDA/cuDNN DLLs are not found.
- If Realtime variant is requested, switch back to `variant="nano"` until the
  Realtime path is implemented and locally validated.
