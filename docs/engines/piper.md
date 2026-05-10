# Piper Engine

`PiperEngine` shells out to the Piper executable and streams raw PCM from a
Piper ONNX voice model. It is a practical local deployment path when you can
ship a Piper binary and model files.

## Install

Install RealtimeTTS with the Piper extra, then set up Piper separately. The
Python extra only covers RealtimeTTS-side dependencies; it cannot install the
Piper executable or voice model files.

```bash
pip install "realtimetts[piper]"
```

Download or build a Piper executable and a Piper voice model. You can pass the
executable path to `PiperEngine` or set `PIPER_PATH`.

```powershell
$env:PIPER_PATH = "D:\path\to\piper.exe"
```

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, PiperEngine, PiperVoice


if __name__ == "__main__":
    voice = PiperVoice(
        model_file="voices/en_US-lessac-medium.onnx",
        config_file="voices/en_US-lessac-medium.onnx.json",
    )
    engine = PiperEngine(voice=voice)
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Piper.")
    stream.play()
```

## Source Notes

- `PiperVoice(model_file, config_file=None)` derives `config_file` from
  `model_file + ".json"` when possible.
- `PiperEngine(piper_path=None, voice=None, debug=False)` uses `PIPER_PATH` and
  then `piper.exe` when `piper_path` is omitted.
- The engine runs Piper with `--output-raw` and feeds text through stdin.
- Sample rate is read from the voice JSON config; fallback is 16000 Hz.
- `get_voices()` returns an empty list because Piper voices are local files, not
  discoverable through the wrapper.

## Troubleshooting

- `Piper executable not found`: set `PIPER_PATH` or pass `piper_path`.
- If speech plays at the wrong speed, check that the matching `.json` config is
  passed with the ONNX model.
