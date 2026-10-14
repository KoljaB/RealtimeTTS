# Breeze TTS 2

BreezeEngine streams mono signed PCM16 at 24 kHz from a local
[HoppouAI/Breeze-TTS-2.cpp](https://github.com/HoppouAI/Breeze-TTS-2.cpp) worker.
The worker owns the GPU and cached voice references; the engine implements
RealtimeTTS audio queues, voice management, native pause/resume and cancellation.
It does not load a second model.

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

**The native package has not been published yet.** Build instructions are in
`native/breeze/README.md` in the source repository. With matching local RealtimeTTS and Windows native
wheels, install and start it using:

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
