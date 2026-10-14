# CTC Forced Word Alignment

Optional CTC forced alignment lets RealtimeTTS emit `on_word` callbacks for engines that do not provide native word timings.

This is not ASR transcription: the text is already known, and the aligner uses the generated audio plus the known sentence text to find word times.

Install a supported engine and, where the upstream omniASR stack supports your platform, the optional backend:

```bash
pip install "realtimetts[system,omniasr]"
```

On Windows with Python 3.12, the current `omnilingual-asr` dependency resolver can fail while resolving `fairseq2` and old `networkx`/`decorator` builds. That is an upstream packaging blocker for the direct Python backend, not normal ASR behavior and not a RealtimeTTS timing-queue issue. The RealtimeTTS alignment interface remains ready for an ONNX/runtime backend exported from the same `facebook/omniASR-CTC-300M` model.

Enable alignment on `TextToAudioStream`:

```python
from RealtimeTTS import TextToAudioStream, OpenAIEngine


def on_word(timing):
    print(timing.word, timing.start_time, timing.end_time)


engine = OpenAIEngine(response_format="pcm")
stream = TextToAudioStream(
    engine,
    on_word=on_word,
    align_words=True,
)

stream.feed("Hello world. This sentence is aligned separately.").play()
```

To align an existing WAV file and transcript to character timestamps:

```python
from RealtimeTTS.alignment import align_wav_transcript_characters

characters = align_wav_transcript_characters(
    "speech.wav",
    "hello world",
    channel=0,  # optional: select a single channel from stereo dialogue audio
)

for item in characters:
    print(item.index, item.character, item.start_time, item.end_time, item.score)
```

The default file aligner uses torchaudio's `MMS_FA` acoustic model and
tokenizer, then runs RealtimeTTS's vectorized CTC alignment. It does not call
the deprecated/removed `torchaudio.functional.forced_align` or
`bundle.get_aligner` APIs. The MMS model is unchanged. Character indices include
normalized whitespace positions, and timings are relative to the WAV start.

To force the omniASR CTC backend with VAD-gated non-blank tokens:

```python
characters = align_wav_transcript_characters(
    "speech.wav",
    "hello world",
    channel=0,
    backend="omniasr",
    use_vad=True,
    vad_tolerance_seconds=0.05,
)
```

That backend gates non-blank CTC tokens to Silero VAD speech regions with 50 ms
tolerance. For noisy stereo dialogue, it does a second pass that allows
low-energy supplemental speech regions only after the first VAD-gated coarse
start, which avoids treating earlier channel bleed as the speaker's transcript.

The default backend is `facebook/omniASR-CTC-300M`. The model loads lazily the first time alignment is needed. Engines that already publish native word timings keep using their native timing path.

For an interactive terminal demo that streams text and highlights the currently spoken word with native timings:

```bash
python tests/ctc_word_highlight_demo.py --engine kokoro
```

To exercise CTC forced alignment instead:

```bash
python tests/ctc_word_highlight_demo.py --engine system --force-align
```

The demo waits for per-sentence alignment by default so the first model load/download does not race ahead of playback. After the model is warm, use `--nonblocking` to test the production-style path where alignment does not delay playback.
