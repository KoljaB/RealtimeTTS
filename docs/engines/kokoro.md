# Kokoro Engine

`KokoroEngine` wraps the Kokoro pipeline. It is a local neural engine with
language-code-aware voice selection, voice formulas, and timing support.

## Install

```bash
pip install "realtimetts[kokoro]"
```

For extra language stacks, install the matching RealtimeTTS extras:

```bash
pip install "realtimetts[kokoro,jp,zh,ko]"
```

Some non-English paths may also require OS-level espeak-ng support depending on
the Kokoro package and platform.

## Minimal Use

```python
from RealtimeTTS import TextToAudioStream, KokoroEngine


if __name__ == "__main__":
    engine = KokoroEngine(voice="af_heart")
    stream = TextToAudioStream(engine)
    stream.feed("Hello from Kokoro.")
    stream.play()
```

## Source Notes

- Default voice is `af_heart`.
- Voice prefixes determine language codes: `af`/`am` American English,
  `bf`/`bm` British English, `jf`/`jm` Japanese, `zf`/`zm` Mandarin Chinese,
  plus Spanish, French, Hindi, Italian, and Brazilian Portuguese prefixes.
- Weighted voice formulas such as `0.3*af_sarah + 0.7*am_adam` are parsed and
  cached.
- Important controls include `default_speed`, silence trimming fields, and
  `debug`.
- Zaphod dev-log benchmarks found Kokoro among the fastest balanced local
  baselines in that environment.

## Troubleshooting

- If a voice loads with the wrong language pipeline, pass a `KokoroVoice` with
  an explicit `language_code`.
- Missing language package errors usually mean the corresponding `jp`, `zh`, or
  `ko` extras were not installed.
