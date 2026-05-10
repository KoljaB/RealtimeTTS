# StyleTTS Engine

`StyleTTSEngine` wraps a local StyleTTS/StyleTTS2 checkout. It is a local neural
reference-audio engine and requires the upstream repository plus model assets.

## Install

Install RealtimeTTS with the StyleTTS extra, then set up the StyleTTS
repository separately. The extra covers the Python dependencies RealtimeTTS can
declare; it does not provide the upstream checkout, model config, checkpoint, or
reference audio.

```bash
pip install "realtimetts[styletts]"
```

Follow the upstream StyleTTS installation for repository-specific setup, model
config, checkpoint, phonemizer, and reference audio requirements.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, StyleTTSEngine, StyleTTSVoice


if __name__ == "__main__":
    voice = StyleTTSVoice(
        model_config_path="StyleTTS2/Models/LJSpeech/config.yml",
        model_checkpoint_path="StyleTTS2/Models/LJSpeech/epoch_2nd_00100.pth",
        ref_audio_path="reference.wav",
    )
    engine = StyleTTSEngine(style_root="D:/path/to/StyleTTS2", voice=voice)
    stream = TextToAudioStream(engine)
    stream.feed("Hello from StyleTTS.")
    stream.play()
```

## Source Notes

- `style_root` is inserted into `sys.path`; pass the root of the upstream
  checkout.
- `StyleTTSVoice` requires a model config path, checkpoint path, and reference
  audio path.
- Key synthesis controls are `alpha`, `beta`, `diffusion_steps`,
  `embedding_scale`, `seed`, and silence duration controls.
- Source imports include `yaml`, `torch`, `torchaudio`, `librosa`, `nltk`,
  `munch`, StyleTTS model modules, and `phonemizer`.
- Output is mono 16-bit PCM at 24000 Hz.

## Troubleshooting

- Import failures usually mean `style_root` is not the upstream repository root
  or its dependencies are not installed in the active environment.
- StyleTTS phonemizer setups often need espeak/espeak-ng installed at the OS
  level.
