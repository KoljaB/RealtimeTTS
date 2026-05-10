# Engine Selection

RealtimeTTS supports local engines, cloud engines, free-service wrappers, and
experimental local voice-cloning stacks. Pick the smallest engine that matches
your latency, quality, licensing, and deployment needs.

Status on this page is documentation status, not a guarantee of support level.
Each current concrete engine source now has a focused page under `docs/engines/`.

## Fast Choices

| Need | Start with |
| --- | --- |
| Verify local audio quickly | `SystemEngine` |
| Free simple service, no API key | `GTTSEngine` or `EdgeEngine` |
| Commercial cloud API | `OpenAIEngine`, `AzureEngine`, `ElevenlabsEngine`, `CartesiaEngine`, `MiniMaxEngine`, `CambEngine`, or `ModelsLabEngine` |
| Word timings | `AzureEngine`; `KokoroEngine` for supported English voices |
| Local neural voice cloning | `CoquiEngine`, `ZipVoiceEngine`, `OmniVoiceEngine`, `FasterQwenEngine`, `LuxTTSEngine`, `ChatterboxEngine`, `SoproTTSEngine`, `PocketTTSEngine`, `NeuTTSEngine`, or `MossTTSEngine` |
| Reliability fallback | Pass a list of compatible engines to `TextToAudioStream` |

## Engine Matrix

| Engine | Type | Install path observed | Needs | Output/playback notes |
| --- | --- | --- | --- | --- |
| [`SystemEngine`](engines/system.md) | Local system TTS | `realtimetts[system]` | System voices | PCM through PyAudio. |
| [`GTTSEngine`](engines/gtts.md) | Free service wrapper | `realtimetts[gtts]` | Network access | Service-backed behavior; no API key. |
| [`EdgeEngine`](engines/edge.md) | Free service wrapper | `realtimetts[edge]` | Network access, `mpv` | Compressed audio path through mpv. |
| [`OpenAIEngine`](engines/openai.md) | Cloud API | `realtimetts[openai]` | `OPENAI_API_KEY` | MP3 uses mpv; PCM uses PyAudio. |
| [`AzureEngine`](engines/azure.md) | Cloud API | `realtimetts[azure]` | Constructor key and region | PCM through PyAudio; word timings supported. |
| [`ElevenlabsEngine`](engines/elevenlabs.md) | Cloud API | `realtimetts[elevenlabs]` | `ELEVENLABS_API_KEY`, `mpv` | MPEG audio path through mpv. |
| [`CambEngine`](engines/camb.md) | Cloud API | `realtimetts[camb]` | `CAMB_API_KEY` | API-backed multilingual synthesis. |
| [`MiniMaxEngine`](engines/minimax.md) | Cloud API | `realtimetts[minimax]` | `MINIMAX_API_KEY`, `mpv` | MP3 output, pytest coverage exists. |
| [`CartesiaEngine`](engines/cartesia.md) | Cloud API | `realtimetts[cartesia]` | `CARTESIA_API_KEY` | Raw PCM formats in source. |
| [`ModelsLabEngine`](engines/modelslab.md) | Cloud API | `realtimetts[modelslab]` | `MODELSLAB_API_KEY`, `mpv` | Present in engines package, not root export. |
| [`CoquiEngine`](engines/coqui.md) | Local neural | `realtimetts[coqui]` | Local model files, likely GPU | Voice cloning; call `shutdown()` in scripts. |
| [`PiperEngine`](engines/piper.md) | Local executable | `realtimetts[piper]` plus Piper binary | Piper binary, model/config files | Very practical for local deployment; uses `PIPER_PATH`. |
| [`StyleTTSEngine`](engines/styletts.md) | Local neural | `realtimetts[styletts]` plus StyleTTS2 checkout | StyleTTS2 checkout/model assets | Local voice/reference-audio workflow. |
| [`ParlerEngine`](engines/parler.md) | Local neural | `realtimetts[parler]` | Parler stack, GPU recommended | Heavy local model setup. |
| [`KokoroEngine`](engines/kokoro.md) | Local neural | `realtimetts[kokoro]` plus language extras | Optional `jp`, `zh`, `ko` extras | Word timings for supported English voices. |
| [`OrpheusEngine`](engines/orpheus.md) | Local/API-style | `realtimetts[orpheus]` | OpenAI-compatible endpoint or local model stack | LM Studio path documented from Zaphod notes. |
| [`FasterQwenEngine`](engines/faster-qwen.md) | Local neural | `realtimetts[qwen]` | Faster Qwen3 stack, reference audio | Voice cloning and emotion examples exist. |
| [`OmniVoiceEngine`](engines/omnivoice.md) | Local neural | `realtimetts[omnivoice]` | OmniVoice stack, reference audio/text | Multilingual voice cloning. |
| [`PocketTTSEngine`](engines/pockettts.md) | Local lightweight | `realtimetts[pockettts]` | `pocket-tts` package | CPU-oriented voice cloning path. |
| [`NeuTTSEngine`](engines/neutts.md) | Local neural | `realtimetts[neutts]`, optional `neutts-gguf` | `neutts[llama,onnx]` for GGUF streaming | Voice cloning with reference audio. |
| [`ZipVoiceEngine`](engines/zipvoice.md) | Local neural | `realtimetts[zipvoice]` plus checkout | ZipVoice checkout/assets or Docker example | Use distill with at least 3 steps for fast quality work. |
| [`LuxTTSEngine`](engines/luxtts.md) | Local neural | `realtimetts[luxtts]` | LuxTTS stack and model assets | Full-waveform local voice cloning. |
| [`ChatterboxEngine`](engines/chatterbox.md) | Local neural | `realtimetts[chatterbox]` | `chatterbox-tts`, reference WAV | Full-waveform prompt-audio workflow. |
| [`SoproTTSEngine`](engines/sopro.md) | Local neural | `realtimetts[sopro]` | Sopro package, optional HF cache | Native streaming reference-audio workflow. |
| [`SopranoEngine`](engines/soprano.md) | Local neural | `realtimetts[soprano]` | Soprano package/model assets | Single-voice English streaming. |
| [`MossTTSEngine`](engines/moss-tts.md) | Local neural | `realtimetts[moss]` plus runtime assets | MOSS-TTS-Nano assets, CUDA/CUDNN depending on backend | Nano ONNX/torch paths documented; Realtime variant blocked. |

## Fallback Engines

You can pass a list of engines. RealtimeTTS starts with the first one and can
switch when synthesis fails.

```python
from RealtimeTTS import TextToAudioStream, SystemEngine, GTTSEngine


if __name__ == "__main__":
    stream = TextToAudioStream([GTTSEngine(), SystemEngine()])
    stream.feed("Try one engine, then fall back if needed.")
    stream.play()
```

Fallbacks work best when engines accept compatible text, voices, and output
expectations. Avoid mixing engines that require unrelated voice objects unless
you handle voice selection per engine.

## Licensing

The RealtimeTTS source is MIT licensed, but engine providers, model weights,
voice data, and generated audio can carry separate restrictions. Check provider
terms before commercial use.
