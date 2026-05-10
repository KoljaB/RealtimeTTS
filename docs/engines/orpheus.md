# Orpheus Engine

`OrpheusEngine` talks to an OpenAI-compatible completions endpoint and decodes
Orpheus audio tokens. In the Zaphod dev-log this was run through LM Studio.

## Install

```bash
pip install "realtimetts[orpheus]"
```

The `orpheus` extra installs `snac`, but you still need a running compatible
model server. Source defaults:

```text
api_url = http://127.0.0.1:1234/v1/completions
model = orpheus-3b-0.1-ft
```

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, OrpheusEngine, OrpheusVoice


if __name__ == "__main__":
    engine = OrpheusEngine(voice=OrpheusVoice("tara"))
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Orpheus.")
    stream.play()
```

## Source Notes

- Available source voice names are `tara`, `leah`, `jess`, `leo`, `dan`, `mia`,
  `zac`, and `zoe`.
- `temperature`, `top_p`, `max_tokens`, and `repetition_penalty` are sent to the
  completions endpoint.
- The prompt format is `<|audio|>{voice}: {text}<|eot_id|>`.
- Output is mono 16-bit PCM at 24000 Hz.

## Zaphod Dev-Log Notes

- The working smoke path required LM Studio at `127.0.0.1:1234` with
  `pkmx/orpheus-3b-0.1-ft`.
- Benchmarks showed Orpheus was usable but not among the fastest local paths in
  that environment.

## Troubleshooting

- If synthesis returns no audio, verify the completions endpoint, model name,
  and streaming response format.
- The engine has two `synthesize` definitions in source; the later definition
  is the active one in Python.
