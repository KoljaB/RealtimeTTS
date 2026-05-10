# SoproTTS Engine

`SoproTTSEngine` wraps SoproTTS. It is a local reference-audio engine with a
native streaming API, making it one of the more practical low-latency cloning
paths in the Zaphod notes.

## Install

```bash
pip install "realtimetts[sopro]"
```

For CUDA, install a matching PyTorch and torchaudio build before or after the
Sopro package, then run `pip check`.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, SoproTTSEngine, SoproTTSVoice


if __name__ == "__main__":
    voice = SoproTTSVoice(ref_audio_path="reference.wav")
    engine = SoproTTSEngine(device="cuda", voice=voice)
    stream = TextToAudioStream(engine)
    stream.feed("Hello from SoproTTS.")
    stream.play()
    engine.shutdown()
```

## Source Notes

- `SoproTTSVoice` accepts `ref_audio_path`, `ref_audio`, or `prompt_wav_path`.
  The source ignores `ref_text` and `language`.
- Defaults include `model_name="samuel-vitorino/sopro"`, `device="cuda"`,
  `max_frames=400`, `top_p=0.9`, `temperature=1.05`, `anti_loop=True`,
  `ref_seconds=12.0`, and `chunk_frames=6`.
- Passing `cache_dir` sets both `HF_HOME` and `HF_HUB_CACHE` before model load,
  because the Sopro Mimi codec uses Hugging Face environment cache settings.
- Reference audio is prepared and cached by path, mtime, and `ref_seconds`.
- Output is mono 16-bit PCM at Sopro's `TARGET_SR`.

## Zaphod Dev-Log Notes

- Working install used `sopro==1.5.0`, `torch==2.11.0+cu128`,
  `torchaudio==2.11.0+cu128`, `transformers==5.8.0`,
  `huggingface_hub==1.14.0`, and `soundfile==0.13.1`.
- A sandboxed first load failed when Hugging Face was unreachable; retry with
  approved network passed.
- NLTK `punkt_tab` needed to be copied into the Sopro venv for local smoke tests.
- Default `chunk_frames=6` gave good TTFA/abort balance in the benchmark notes;
  `chunk_frames=3` improved TTFA but worsened RTF and abort jitter.
- Official notes and Zaphod listening notes both warn that cloning quality can
  be inconsistent; listen to samples before making it a default.

## Troubleshooting

- If offline runs still call Hugging Face, set both `HF_HOME` and
  `HF_HUB_CACHE` or pass `cache_dir`.
- If startup reports missing NLTK `punkt_tab`, install or copy that tokenizer
  data into the active venv.
