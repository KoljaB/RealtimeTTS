# Output And Files

RealtimeTTS can play audio locally, write a WAV file, or pass raw chunks to your
application.

## Local Playback

PCM engines use PyAudio and `output_device_index`:

```python
stream = TextToAudioStream(engine, output_device_index=3)
stream.feed("Speak through a selected PyAudio device.")
stream.play()
```

Use PyAudio device inspection tools or a small local script to find device
indices for your machine.

Compressed-output engines use mpv. Examples include Edge, ElevenLabs, OpenAI
when using MP3, MiniMax, and ModelsLab.

```python
stream = TextToAudioStream(engine, mpv_audio_device="auto")
```

Run this in a terminal to list mpv audio devices:

```bash
mpv --audio-device=help
```

## Muted Mode

Muted mode disables local speaker playback but still lets you write files or
process chunks.

```python
stream = TextToAudioStream(engine, muted=True)
stream.feed("Generate audio without playing it.")
stream.play(output_wavfile="speech.wav")
```

You can also pass `muted=True` to a single `play()` call.

## WAV Output

```python
stream.feed("Save this line.")
stream.play(output_wavfile="speech.wav", muted=True)
```

The output file is opened when playback begins and finalized when playback
finishes. For long-running applications, create a new output path per response.

## Audio Chunk Callback

Use `on_audio_chunk` when another process needs audio bytes immediately.

```python
def handle_chunk(chunk: bytes):
    print("chunk bytes:", len(chunk))


stream.feed("Send chunks to my application.")
stream.play(on_audio_chunk=handle_chunk, muted=True)
```

Chunk format depends on the engine. Use `engine.get_stream_info()` to inspect
the active engine's format, channel count, and sample rate.

## Buffer And Chunk Tuning

`TextToAudioStream` accepts audio playback sizing controls:

```python
stream = TextToAudioStream(
    engine,
    frames_per_buffer=1024,
    playout_chunk_size=-1,
)
```

Smaller chunks can reduce latency but increase CPU overhead. Larger chunks can
smooth playback but add delay. Keep defaults until you have a concrete latency
or stuttering problem.

## Volume

The stream exposes clamped volume control from `0.0` to `1.0`:

```python
stream.volume = 0.7
# or
stream.set_volume(0.7)
```

Volume scaling applies inside the playback path. Some compressed or external
playout behavior can still depend on the engine and mpv path.
