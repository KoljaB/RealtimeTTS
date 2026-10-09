# Breeze TTS 2

BreezeEngine streams mono signed PCM16 at 24 kHz from a local
[HoppouAI/Breeze-TTS-2.cpp](https://github.com/HoppouAI/Breeze-TTS-2.cpp) worker.
The worker owns the GPU and cached voice references; the engine implements
RealtimeTTS audio queues, voice management, native pause/resume and cancellation.
It does not load a second model.

## Installation and worker

Install the optional server dependencies with:

~~~text
python -m pip install "realtimetts[breeze-server]>=0.9.0rc1"
~~~

For source development, use `python -m pip install -e ".[breeze-server]"`.
The native worker and model assets are separate. The native preparation commands
below use the matching RealtimeTTS source checkout or unpacked source distribution,
which includes both `tools/prepare_breeze_cpp.py` and `tools/breeze_cpp_controls.json`.

Add the playback extra when using TextToAudioStream with local audio devices.
The new entry point is realtimetts-breeze-server; source execution also works:

~~~text
python -m RealtimeTTS.breeze_server
~~~

Use Hoppou commit a0e177f91242ebbd2ea8d61548d742c89e4b9e06 (v0.1.0),
including its ggml submodule. Prepare the native control patch before building:

~~~text
python tools/prepare_breeze_cpp.py path/to/Breeze-TTS-2.cpp
cmake -S path/to/Breeze-TTS-2.cpp -B path/to/build -DBREEZE_VULKAN=ON -DBREEZE_MP3=OFF
cmake --build path/to/build --config Release --target breeze-server
~~~

Use upstream platform build instructions for the compiler and Vulkan SDK.
The supplied patch is idempotent and validates exact anchors. It adds checkpoints
between GPU stages, acknowledged pause/resume, cancelled completion, and first-token
timing. This is necessary: stock v0.1.0 only cancels at audio-chunk callbacks.
The engine refuses an unpatched worker instead of silently claiming fast controls.

Start the worker with the Q8 GGUF and its adjacent audio_tokenizer assets:

~~~text
breeze-server breeze-tts-2-q8_0.gguf --host 127.0.0.1 --port 8137 --ws-port 8138 --chunk-first 1 --chunk-max 40 --voices-dir native-voices
~~~

Then start the compatible API:

~~~text
python -m RealtimeTTS.breeze_server --host 127.0.0.1 --port 18084 --voice my_voice --voice-reference "reference.wav" --voice-reference-text "The exact reference transcript." --voice-dir api-voices
~~~

CFG defaults to 4; the seed defaults to 44. The full reference is encoded and
cached by audio/transcript hash. A changed reference or transcript gets a new cache
key. Cold shaders and new prompt shapes can make initial requests slower.
Non-loopback API binding requires an environment-based bearer key.

## Engine API

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
CPU detector for `POST /v1/text/languages`. The `breeze-server` extra includes its
fastText dependency. The model is warmed at startup; classification runs outside
the synthesis lock. This does not change Breeze's supported speech languages.
