# Breeze TTS 2

BreezeEngine streams mono signed PCM16 at 24 kHz from a local
[HoppouAI/Breeze-TTS-2.cpp](https://github.com/HoppouAI/Breeze-TTS-2.cpp) worker.
The worker owns the GPU and cached voice references; the engine implements
RealtimeTTS audio queues, voice management, native pause/resume and cancellation.
It does not load a second model.

## Bundled reference voices

The wheel and source distribution include the selected English and Chinese reference WAVs and their exact transcripts:

```python
from RealtimeTTS import BreezeEngine, BreezeVoice, TextToAudioStream

# Connect to the running local Breeze worker described below.
engine = BreezeEngine(voice=BreezeVoice.builtin("en"))
TextToAudioStream(engine).feed("Good morning!").play()
```

Use `BreezeVoice.builtin("zh")` for Chinese. Full language names and bundled names (`breeze-en`, `breeze-zh`) also work. `engine.get_voices()` lists registered worker voices followed by the bundled references; selecting a bundled voice uploads its packaged WAV and transcript to the worker using the existing voice cache. No separate audio download or local asset path is required.

The Chinese 400-case result below used explicit read-once instructions and a per-request token cap. The factory selects the reference recording; it does not change synthesis instructions or token limits.

## Current reference files and Breeze evaluation (2026-10-19)

The table lists the current selected Qwen reference files across all ten languages. Breeze results are available for English and Chinese only. The Qwen 0.6B files are bundled as Breeze voices; the Qwen 1.7B files form the separate Qwen reference set.

| Language | Selected Qwen reference | Duration | Qwen GPU result | Qwen CPU result | Breeze on selected reference | Additional Breeze-tested candidate |
|---|---|---:|---|---|---|---|
| English | [qwen17b_en_02](https://github.com/KoljaB/RealtimeTTS/blob/v0.9.0rc4/RealtimeTTS/voices/qwen/qwen17b_en_02.wav) | 16.80 s | Not tested | 400/400 clear | 398/400 clear; noise 1, silence 1; two >4 s tails (one unflagged) | [qwen06_en_03](https://github.com/KoljaB/RealtimeTTS/blob/v0.9.0rc4/RealtimeTTS/voices/breeze/qwen06_en_03.wav), 13.215 s: 400/400 clear, no >4 s tails |
| Spanish | [qwen17b_es_e2_40](https://github.com/KoljaB/RealtimeTTS/blob/v0.9.0rc4/RealtimeTTS/voices/qwen/qwen17b_es_e2_40.wav) | 18.24 s | Not tested | 400/400 clear | Not tested | - |
| German | [qwen17b_de_37](https://github.com/KoljaB/RealtimeTTS/blob/v0.9.0rc4/RealtimeTTS/voices/qwen/qwen17b_de_37.wav) | 17.76 s | 395/400 clear; noise 3, silence 2 | 102/103 clear; noise 1, silence 0 | Not tested | - |
| French | [qwen17b_fr_31](https://github.com/KoljaB/RealtimeTTS/blob/v0.9.0rc4/RealtimeTTS/voices/qwen/qwen17b_fr_31.wav) | 17.92 s | 394/400 clear; noise 2, silence 4 | 96/100 clear; noise 0, silence 4 | Not tested | - |
| Portuguese | [qwen17b_pt_11](https://github.com/KoljaB/RealtimeTTS/blob/v0.9.0rc4/RealtimeTTS/voices/qwen/qwen17b_pt_11.wav) | 15.84 s | 391/400 clear; noise 0, silence 9 | 99/100 clear; noise 0, silence 1 | Not tested | - |
| Russian | [qwen17b_ru_09](https://github.com/KoljaB/RealtimeTTS/blob/v0.9.0rc4/RealtimeTTS/voices/qwen/qwen17b_ru_09.wav) | 19.12 s | 400/400 clear | 100/100 clear | Not tested | - |
| Italian | [qwen17b_it_03](https://github.com/KoljaB/RealtimeTTS/blob/v0.9.0rc4/RealtimeTTS/voices/qwen/qwen17b_it_03.wav) | 15.20 s | 99/100 clear; noise 0, silence 1 | 98/100 clear; noise 1, silence 1 | Not tested | - |
| Japanese | [qwen17b_ja_09](https://github.com/KoljaB/RealtimeTTS/blob/v0.9.0rc4/RealtimeTTS/voices/qwen/qwen17b_ja_09.wav) | 13.84 s | 87/100 clear; noise 4, silence 9 | 87/100 clear; noise 4, silence 9 | Not tested | - |
| Chinese | [qwen17b_zh_01](https://github.com/KoljaB/RealtimeTTS/blob/v0.9.0rc4/RealtimeTTS/voices/qwen/qwen17b_zh_01.wav) | 16.24 s | 400/400 clear | 100/100 clear | 385/400 clear; noise 4, silence 11; 29 >4 s tails (21 unflagged) | [qwen06_zh_03](https://github.com/KoljaB/RealtimeTTS/blob/v0.9.0rc4/RealtimeTTS/voices/breeze/qwen06_zh_03.wav), 12.335 s: 400/400 clear, no >4 s tails |
| Korean | [qwen17b_ko_08](https://github.com/KoljaB/RealtimeTTS/blob/v0.9.0rc4/RealtimeTTS/voices/qwen/qwen17b_ko_08.wav) | 13.44 s | 98/100 clear; noise 0, silence 2 | 96/100 clear; noise 2, silence 2 | Not tested | - |

All Breeze runs used the same 100 texts with four seeds (42, 43, 69 and 177), Breeze TTS 2 Q8_0 on Vulkan, and the automatic noise/silence detectors. "Clear" means no acoustic detector finding; it does not mean a human confirmed every clip. The >4 s tail check is tracked separately: the selected English reference produced two such overruns, one with no acoustic flag; the selected Chinese reference produced 29, of which 21 had no acoustic flag. Both Qwen 0.6B candidates had zero acoustic flags and no >4 s tails.

Candidate WAVs were synthesized on Windows with Qwen3-TTS 0.6B Base Q8_0 using qwentts.cpp CUDA 12, generation seed 203. English candidate SHA-256: 1e930fa36a8a2e9faf1a0efaae999ac5ce21b9e81546e2278b37fe9ae5daa912; Chinese candidate SHA-256: 2c9524926c6728f5ddf81717f4e42bc8fff48fae06cfd35211411e577cc8b00d. The Chinese Breeze run used the instruction "read exactly once, do not add or repeat anything, then stop" and max_new_tokens=max(16, 5 * character_count). Neither candidate has Qwen CPU qualification recorded here.

## Model and output licensing

The Breeze TTS 2 model and its self-hosted outputs are for
**research and non-commercial use only**. **Commercial use requires a separate
licence from BreezeBlue.** Read the
[BreezeBlue model licence](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE)
and direct commercial licensing enquiries to
[contact@breeze.blue](mailto:contact@breeze.blue).

RealtimeTTS's MIT source-code licence does not grant commercial rights to the
Breeze TTS 2 model or its self-hosted outputs.

## Model and installation

The initial supported configuration is **Breeze TTS 2 Q8_0**, for English and
Mandarin, through the Vulkan worker. Download `breeze-tts-2-q8_0.gguf` from the
[GGUF model repository](https://huggingface.co/HoppouAI/Breeze-TTS-2.cpp).
It is about 3.3 GB and contains the tokenizer and vocoder. An adjacent
`audio_tokenizer/` directory is only needed when converting the original model.

Q8_0 keeps the first release focused on the upstream recommended quantization.
Q4, F16 and experimental `-dd` variants are not qualified by this integration.
CFG stays at **4**, with seed **44**. Lower CFG changes voice guidance and delivery;
it is not an automatic speed optimization.

### Prebuilt worker candidate

The native worker is a separate optional `realtimetts-breeze-native` wheel,
keeping the base RealtimeTTS install unchanged. It packages the patched executable
and licence notices; model weights are downloaded separately. The candidate
targets x64 with AVX2 and a Vulkan-capable GPU driver. Python-side inference does
not need PyTorch or a CUDA toolkit.

The Windows TestPyPI candidate installs the native worker through the
`breeze-server` extra. See the [short uv quickstart](../testpypi-quickstart.md).
For local wheel development, install matching framework and native wheels:

~~~text
python -m pip install "./realtimetts-0.9.0rc2-py3-none-any.whl[breeze-server]" ./realtimetts_breeze_native-0.1.0-py3-none-win_amd64.whl
python -m RealtimeTTS.breeze_server --model /models/breeze-tts-2-q8_0.gguf
~~~

For source development, replace the first package argument with
`-e ".[breeze-server]"`. Linux wheels require separate platform qualification;
do not relabel a Windows or unaudited Linux build.

Open `http://127.0.0.1:18084/studio` and choose Voice Design to start without a
reference WAV. To clone a voice at startup, add `--voice my_voice`,
`--voice-reference reference.wav` and
`--voice-reference-text "The exact reference transcript."`.
A registered reference is cached and reused on later starts.

`--model` starts one owned worker on private loopback ports and stops it when the
API exits. Startup checks the native controls and Vulkan backend. A missing
driver produces an error instead of silently advertising CPU inference as GPU
inference. Cold shaders and new prompt shapes can make first requests slower.

On 10 October 2026, the installed Windows candidate passed GPU synthesis,
reference caching, pause/resume/cancel, reuse after cancellation, and API Voice
Design without a reference WAV. An isolated RTX 2080 SUPER run with Q8_0, CFG 4
and seed 44 measured:

| Request | First PCM | Audio duration | Generation time | Real-time factor |
| --- | ---: | ---: | ---: | ---: |
| Cold short sentence | 391 ms | 1.68 s | 1.78 s | 1.06 |
| Warm repeat | 328 ms | 1.68 s | 1.53 s | 0.91 |
| Longer sentence | 313 ms | 5.52 s | 4.28 s | 0.78 |

A real-time factor below 1 is faster than playback. These three measurements
are a limited hardware-specific check; first PCM excludes playback buffering
and device latency. The cold short request was slower than real time.

### Existing worker or source build

Connecting to a separately running worker needs only the Python extra:

~~~text
python -m pip install "realtimetts[breeze-server]>=0.9.0rc1"
python -m RealtimeTTS.breeze_server --worker-url http://127.0.0.1:8137 --worker-ws-url ws://127.0.0.1:8138
~~~

The matching source checkout or unpacked source distribution includes
`tools/prepare_breeze_cpp.py` and `tools/breeze_cpp_controls.json`.
Use Hoppou commit `a0e177f91242ebbd2ea8d61548d742c89e4b9e06` (v0.1.0), including
its pinned ggml submodule. Prepare the controls before building:

~~~text
git clone --branch v0.1.0 --recursive https://github.com/HoppouAI/Breeze-TTS-2.cpp.git breeze-source
python tools/prepare_breeze_cpp.py breeze-source
cmake -S breeze-source -B breeze-build -DCMAKE_BUILD_TYPE=Release -DBREEZE_VULKAN=ON -DBREEZE_MP3=OFF -DBREEZE_BUILD_SHARED=OFF -DBREEZE_BUILD_CLI=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF
cmake --build breeze-build --config Release --target breeze-server
~~~

Use upstream platform instructions for the compiler and Vulkan SDK. Stock v0.1.0
binaries lack the required native pause/cancel checkpoints and are rejected by
the engine. Start the rebuilt executable with:

~~~text
breeze-server breeze-tts-2-q8_0.gguf --host 127.0.0.1 --port 8137 --ws-port 8138 --chunk-first 1 --chunk-max 40 --voices-dir native-voices
~~~

The small first chunk starts streaming promptly; subsequent chunks grow for
throughput. Reference audio and transcript are cached together; changing either
creates a new cache entry. Add the `playback` extra for local audio devices.
Non-loopback API binding requires an environment-based bearer key.

## Engine API

Connect directly to a separately running native worker (ports 8137/8138 by default):

~~~python
from RealtimeTTS import BreezeEngine, BreezeVoice, TextToAudioStream

engine = BreezeEngine(BreezeVoice(
    "my_voice",
    ref_audio="reference.wav",
    ref_text="The exact reference transcript.",
    instruct="Warm and amused, with natural conversational delivery.",
))
stream = TextToAudioStream(engine)
stream.feed("Well, (chuckle) I did not see that coming.").play()
# From the interruption thread:
# stream.stop()  # discards playback and calls engine.stop()
~~~

engine.pause(timeout=2) waits for a native checkpoint acknowledgement.
engine.resume() continues that same generation. engine.stop() requests
cancellation without waiting for a whole audio chunk or restarting the model.
An executing GPU operation cannot be preempted; a prefill or codec operation is
therefore a lower bound on stop latency. Early and late PCM is fenced by the
request cancellation event.

## Compatible server and directions

The existing realtimetts-qwen-http protocol is retained: health/readiness,
capabilities, voice registration, HTTP speech and WebSocket speech-stream,
including pause/resume/cancel, flush and segmented narration. Qwen's default
behavior remains unchanged. Qwen-specific latent voices, clone_mode and subtalker
parameters are rejected for Breeze.

A WebSocket session begins with:

~~~json
{"type":"config","voice":"my_voice","response_format":"pcm"}
~~~

Send directions independently of speech:

~~~json
{"type":"text","instructions":"Amused and lighthearted","text":"Well, (chuckle) that was unexpected."}
{"type":"text","instructions":"Whispering softly","text":"Now listen closely."}
{"type":"end"}
~~~

Instructions are sticky until replaced. A change flushes pending text under its
previous instruction; it cannot retroactively alter an earlier fragment.
An empty string clears the direction. The end event ends text input; controls
remain accepted until synthesis finishes. Disconnect also cancels native work.

For HTTP, use input, voice, instructions, response_format (pcm or wav).

### Whole-text synthesis in Voice Studio

The Studio page has an opt-in **Text ohne Satzaufteilung synthetisieren**
checkbox below the text editor. It sends `split_sentences: false` in the
WebSocket config, bypassing sentence/fragment splitting for that request.
Audio still streams as soon as the model produces it; pause and cancel remain
available. The default remains sentence splitting for fast incremental input.

For manually streamed text, whole-text synthesis waits for `end` or an explicit
`flush`. Voice, instruction and segment changes still flush the preceding text
before applying the change. HTTP PCM/WAV requests already synthesize the
complete input; the checkbox preserves that behavior. Long inputs remain
subject to the model's token/output limits.

The API advertises this option as `features.sentence_splitting`; Studio hides
the checkbox when connected to an older server that cannot honor it.

The engine also understands paragraph-leading [tone] tags in complete input,
including CR/LF and whitespace immediately after a tag. For incremental model
output, Miep parses these tags before segmentation and sends instruction metadata.

The application vocal-event allowlist is:
(laugh), (chuckle), (giggle), (cry), (sob), (whimper), (groan),
(moan), (sigh), (gasp), (inhale), (exhale), (breathe heavily),
(clear throat), (clears throat), (cough), (sniffle), (lip smack),
(click tongue), (yawn), (sneeze), (hiccup), (burp), (gulp), (gag),
(grunt), (scoff), (snort).
These stay in the speech text. The prompt forbids invented tags. This allowlist
does not guarantee that every sound is reproduced convincingly on every seed.

Breeze's [model card](https://huggingface.co/BreezeBlue/Breeze-TTS-2) documents
English and Mandarin, reference-guided directions and inline vocal events.
This integration does not claim additional trained languages or word alignment.


## Background language classification

Add `--language-id-model /path/to/lid.176.bin` to enable the existing RealtimeTTS
CPU detector for `POST /v1/text/languages`. Install its optional dependency with
`python -m pip install fasttext-predict` first. The model is warmed at startup;
classification runs outside
the synthesis lock. This does not change Breeze's supported speech languages.
