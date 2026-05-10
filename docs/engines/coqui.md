# Coqui Engine

`CoquiEngine` wraps Coqui XTTS for local neural TTS and voice cloning. It uses a
worker process, so scripts should use the normal Windows-safe
`if __name__ == "__main__":` guard and call `shutdown()` when done.

## Install

```bash
pip install "realtimetts[coqui]"
```

The default model is `tts_models/multilingual/multi-dataset/xtts_v2` with
`specific_model="v2.0.2"`. If `local_models_path` is not provided, the source
also checks `COQUI_MODEL_PATH`.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, CoquiEngine


if __name__ == "__main__":
    engine = CoquiEngine(voice="reference.wav", language="en")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Coqui XTTS.")
    stream.play()
    engine.shutdown()
```

## Source Notes

- `voice` can be a string or list of reference WAV filenames. `voices_path`
  can point to the directory containing them.
- Source comments say voice cloning works with 44100 Hz or 22050 Hz mono
  32-bit float WAV files.
- Important tuning fields include `speed`, `stream_chunk_size`,
  `overlap_wav_len`, `temperature`, `top_k`, `top_p`, `use_deepspeed`, and
  `device`.
- `get_stream_info()` reports Coqui's PCM playback path. The worker process
  streams chunks back through a pipe.

## Zaphod Dev-Log Notes

- A dedicated Windows speedup env used Python 3.10.20,
  `torch==2.1.2+cu121`, `torchaudio==2.1.2+cu121`, `coqui-tts==0.27.5`,
  `deepspeed==0.11.2`, `transformers==4.57.3`, and `numpy==1.26.4`.
- NumPy 2.x broke DeepSpeed 0.11.2 because `numpy.BUFSIZE` was removed.
- A Transformers/Torch compatibility shim was needed in that Zaphod experiment
  because torch 2.1.2 did not expose the public pytree API expected by the
  installed Transformers build.
- Because Coqui starts a child process, environment shims such as
  `sitecustomize.py` must be visible inside the engine venv, not only in the
  parent process.

## Troubleshooting

- If the first run is slow, the model may be downloading or loading into the
  worker process.
- If DeepSpeed is enabled, pin dependencies carefully and test `pip check`.
- If synthesis hangs in scripts on Windows, confirm the script has a
  `__main__` guard.
