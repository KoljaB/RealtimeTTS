# LuxTTS Engine

`LuxTTSEngine` wraps LuxTTS through `zipvoice.luxvoice.LuxTTS`. It is a local
voice-cloning engine: every voice is a prompt WAV plus optional prompt text.

## Install

The current `setup.py` includes a `luxtts` extra with Git dependencies and a
larger local stack:

```bash
pip install "realtimetts[luxtts]"
```

For development, the Zaphod pass used a local checkout and passed `lux_root`:

```text
third_party\LuxTTS
third_party\LuxTTS-Gradio
third_party\cake
```

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, LuxTTSEngine, LuxTTSVoice


if __name__ == "__main__":
    voice = LuxTTSVoice(
        prompt_wav_path="reference.wav",
        prompt_text="Exact transcript of the reference audio.",
    )
    engine = LuxTTSEngine(lux_root="third_party/LuxTTS", voice=voice)
    stream = TextToAudioStream(engine)
    stream.feed("Hello from LuxTTS.")
    stream.play()
    engine.shutdown()
```

## Source Notes

- `lux_root` is optional, but if provided it is inserted into `sys.path` before
  importing `zipvoice.luxvoice`.
- Default model path is `YatharthS/LuxTTS`.
- Defaults include `device="cuda"`, `threads=4`, `num_steps=4`,
  `guidance_scale=3.0`, `t_shift=0.5`, `speed=1.0`, and `prompt_duration=5.0`.
- Output sample rate is 48000 Hz, or 24000 Hz when `return_smooth=True`.
- The wrapper disables Transformers torchcodec detection on Windows before Lux
  loads Whisper to avoid DLL/ffmpeg mismatch crashes observed in the Zaphod
  setup.
- LuxTTS returns a full waveform. Stop requests are honored before queueing late
  output, not mid-inference.

## Zaphod Dev-Log Notes

- Working CUDA env used Python 3.11.5, `torch==2.11.0+cu128`, and
  `torchaudio==2.11.0+cu128`.
- The pass installed RealtimeTTS editable, then
  `-r third_party\LuxTTS\requirements.txt`.
- Model files from `YatharthS/LuxTTS` and `openai/whisper-base` were cached
  under `D:\hf_cache\hub`.
- NLTK `punkt_tab` was copied into the Lux venv because an import path tried to
  download it.
- 1-step output was considered distorted; step 2 was the practical faster
  setting in that Zaphod listening pass, while the source default remains 4.
- Do not force-install `k2` into the torch 2.11 CUDA Lux env; the checked k2
  wheel path was incompatible.

## Troubleshooting

- Import errors usually mean LuxTTS is not installed or `lux_root` is not the
  upstream checkout root.
- First run may download LuxTTS, Whisper, tokenizer, and vocoder assets.
