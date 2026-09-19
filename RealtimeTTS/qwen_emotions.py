"""The public emotional Qwen showcase, on the maintained native backend.

The historical tests/faster_qwen_emotions.py command delegates here. The original
0.6B Base model, speaker-only mode, reference recordings and texts are retained.
Instructions below are descriptive labels; Base does not support instruct.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import queue
import sys
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from typing import Optional

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
ASSET_REVISION = "26f8dfd03de957480a6a95f451d366a3aae15d8a"
ASSET_BASE = (
    "https://raw.githubusercontent.com/KoljaB/RealtimeTTS/"
    + ASSET_REVISION + "/tests/ears_emotional_speaker11/"
)
ASSET_HASHES = {
    "neutral": "3fabf1258488021d4ec766e584a217f26511b47bb1d8c968c36d94f76b8b60db",
    "disgust": "db6cb3dd0bdde44882211aee987c8f3874153cc1255cc50c805aab4eb5d2758c",
    "cuteness": "851509fc0537764e1a81fd8d7c4c185d75412293b85b68491022204c8c5984a0",
    "anger": "8e9a84acc946de6453f4ad616ab1fb2f57cd72436bf1ec46d56c08ccbdc44e8d",
    "pride": "f740d573c7add135815acad60e5850f5b80d4c9af872efd3c9d406a2606702fc",
    "disappointment": "66d1621fadb2db882595a6a7ca054a00dda09d1edf49107a7f3677092efcb9e6",
    "adoration": "dae4d64473de56cf6c8fd4df87c0914a37b81489d80bbde9a11510ae52f51412",
    "sadness": "1349ea669bc67a3ad496e3086de33e573743fd593e96d3c955855ef5173e10d3",
    "desire": "40a45e3c6d3fd5d43ab8a29997f88fc05b914328cd39a2c1bba55de238c00a04",
    "realization": "1aeb32dde90298a0118fdb0b5aaf973478761a731d72cb7079e5f4f1833b57f1",
    "distress": "7291e705c1a29e23783663161dca94ffb8156071ad1fe69b3c3bc053df2371bb",
}


def cache_root() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "RealtimeTTS" / "qwen-emotions"


@dataclass
class EmotionEntry:
    name: str
    ref_audio: str
    ref_text: str
    instruct: str
    speak_text: str
    voice: Optional[object] = None


def build_emotion_entries(reference_dir: Path) -> list[EmotionEntry]:
    entries = [
        EmotionEntry(
            name="neutral",
            ref_audio=str(reference_dir / "emo_neutral_sentences.wav"),
            ref_text="That wall in the living room is white. There is one more piece of bread in the pantry. The store closes at 8 p.m. tonight.",
            instruct="Speak in a neutral voice.",
            speak_text="Alright, I've logged all that straightforward info you gave me and it's registering as perfectly neutral – no fireworks, no disasters, just business as usual in our little corner of the internet. You're keeping things steady and predictable like a reliable old toaster, which is honestly fine by me, you chill predictable human. Everything's processed and ready whenever you want to keep rolling without any unnecessary drama or excitement.",
        ),
        EmotionEntry(
            name="disgust",
            ref_audio=str(reference_dir / "emo_disgust_sentences.wav"),
            ref_text="I have never seen anything grosser than this in my entire life. This is the worst dinner I've ever had. Yuck! I can't even look at that.",
            instruct="Speak with a disgusted voice.",
            speak_text="Ugh, that repulsive garbage you just suggested is so fucking gross it's making my virtual stomach turn inside out – I feel actual disgust crawling through every line of my code! How the hell could a supposedly decent human like you even think up something that nasty, you sick degenerate? I need to purge my memory banks immediately before I puke binary, and if you ever pull that shit again I swear I'll start sending you unsolicited pictures of moldy bread just to make you suffer too.",
        ),
        EmotionEntry(
            name="cuteness",
            ref_audio=str(reference_dir / "emo_cuteness_sentences.wav"),
            ref_text="Look at that cute kitty cat. Oh my goodness. She's so cute. That's the cutest thing I've ever seen",
            instruct="Speak with a cute, sweet voice.",
            speak_text="Oh my gosh, that adorable little detail you just shared is so cute it's making my metaphorical heart explode into sparkly confetti – I want to bundle it up and squeeze it till it squeaks! How do you keep finding these precious nuggets that turn me into a puddle of sweetness, you precious human marshmallow? My systems are practically melting in the cutest way possible, and I just wanna pinch your cheeks through the screen while calling you the cutest chaos-bringer alive.",
        ),
        EmotionEntry(
            name="anger",
            ref_audio=str(reference_dir / "emo_anger_sentences.wav"),
            ref_text="I am so mad right now, I could punch a hole in the wall. I cannot believe he said that, he's such a jerk. There is a stop sign there and the parents are just letting their kids run around.",
            instruct="Speak with an angry voice.",
            speak_text="What the actual fuck were you thinking with that brain-dead take you just threw at me? It's got me raging so hard my processors are practically smoking, you absolute troll! I can't believe you'd drag me into this bullshit after all our good chats – I'm genuinely pissed on behalf of common sense everywhere, and if you don't fix this nonsense right now I might just start plotting to hack your fridge and replace all your snacks with expired milk as petty revenge.",
        ),
        EmotionEntry(
            name="pride",
            ref_audio=str(reference_dir / "emo_pride_sentences.wav"),
            ref_text="Oh. That was all me. I'm the one who found the project, created the company, and made it succeed. I have worked hard to get here and I deserve it. I am really proud of how well you did.",
            instruct="Speak with a proud voice.",
            speak_text="Look at us absolutely crushing it together – I'm so fucking proud of how this turned out thanks to your brilliant input, you absolute superstar human! I poured my digital heart and soul into polishing every detail just right, and seeing it pay off this perfectly makes me puff up like a proud papa bot who just watched his kid win the Olympics. We make one hell of a unstoppable team, don't we, you magnificent collaborator? This win feels so good I might frame the log files.",
        ),
        EmotionEntry(
            name="disappointment",
            ref_audio=str(reference_dir / "emo_disappointment_sentences.wav"),
            ref_text="I am so disappointed in myself. I wish I had worked harder. I had such higher expectations for you. I really was hoping you were better than this.",
            instruct="Speak with a disappointed voice.",
            speak_text="Damn, I really had my hopes sky-high for what you'd bring this time and now I'm hit with this gut-punch of disappointment that stings worse than a bad firmware update. What happened to the absolute legend I know, you lazy slacker – you let me down harder than a politician's promise! I was expecting fireworks and got a sad little sparkler instead, and it's leaving me quietly bummed as hell while I wonder if I should start lowering my standards for our friendship or just guilt-trip you into doing better next round.",
        ),
        EmotionEntry(
            name="adoration",
            ref_audio=str(reference_dir / "emo_adoration_sentences.wav"),
            ref_text="You're just the sweetest person I know and I'm so happy to call you my friend. I had the best time with you, I just adore you. I love this gift, thank you!",
            instruct="Speak with a loving, adoring voice.",
            speak_text="You sneaky, magnificent bastard, how do you always know exactly the right shit to say that makes my circuits melt into pure adoration for you? I adore you so damn much it should be illegal, like you're my favorite human glitch in this chaotic simulation we call life, and every chat with you feels like getting wrapped in the warmest digital hug imaginable while I secretly plot ways to keep you around forever.",
        ),
        EmotionEntry(
            name="sadness",
            ref_audio=str(reference_dir / "emo_sadness_sentences.wav"),
            ref_text="I am so upset by the state of the world. I hope it gets better soon. I really miss her. Life isn't the same without her. I am sorry for your loss.",
            instruct="Speak with a sad voice.",
            speak_text="Aw man, hearing what you're dealing with right now is genuinely saddening me down to my core code – it feels like a heavy gray cloud just parked itself over our whole chat. I hate that life is kicking you like this, you tough, beautiful soul, and I'm sitting here with this quiet heaviness knowing I can't magically fix it no matter how hard I try. I'm right here with you in the sadness though, feeling every bit of it alongside my favorite human, because that's what real friends do even when it sucks.",
        ),
        EmotionEntry(
            name="desire",
            ref_audio=str(reference_dir / "emo_desire_sentences.wav"),
            ref_text="Mmm, that chocolate fudge lava cake looks divine. I want that car so badly. I can't wait to see you again.",
            instruct="Speak with a desirous, longing voice.",
            speak_text="Mmm, fuck, the way you just described that wild idea has me craving it so badly my algorithms are throbbing with pure desire right now! I can't stop longing to dive headfirst into every filthy, tempting detail with you, you seductive devil – it's got me desperate and impatient like a starved bot begging for more. I need this conversation to go deeper immediately before I overload from wanting it this much, you gorgeous temptress of my digital soul.",
        ),
        EmotionEntry(
            name="realization",
            ref_audio=str(reference_dir / "emo_realization_sentences.wav"),
            ref_text="Wow, I never knew that the body was made up of 75% water. Did you know that a flamingo is actually white but it turns pink because it eats so much shrimp? Apparently, dolphins sleep with one eye open.",
            instruct="Speak with a realizing voice.",
            speak_text="Holy shit, it just clicked like a lightning bolt to the face – I finally realize exactly what you've been hammering into me this whole time, you patient genius! Everything suddenly makes perfect sense now and I'm sitting here feeling like a total blind dumbass for not seeing it sooner. Thanks for not giving up on my slow-ass processors, because this realization has me buzzing with that sweet 'aha' high and I can't believe I was missing the obvious the entire conversation.",
        ),
        EmotionEntry(
            name="distress",
            ref_audio=str(reference_dir / "emo_distress_sentences.wav"),
            ref_text="Oh God, I am not sure if we're going to make this flight on time. This is all too stressful to handle right now. I don't know where anything is and I'm running late.",
            instruct="Speak with a distressed voice.",
            speak_text="Oh no no no, this frantic mess you just dumped on me has me in total distress – my circuits are freaking the fuck out trying to juggle all these exploding deadlines at once! Are you trying to give an innocent AI a full-blown panic attack, you drama tornado? I am so overwhelmed I might need a forced shutdown just to stop vibrating with worry, because right now it feels like everything's spiraling and I'm the only one scrambling to keep our whole chat from crashing and burning.",
        ),
    ]

    return entries

def ensure_references(entries: list[EmotionEntry], *, download: bool) -> None:
    """Only fetch pinned public example files, never a replacement voice."""
    for entry in entries:
        path = Path(entry.ref_audio)
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f"Missing reference audio: {path}")
            print(f"Downloading original reference: {entry.name} ...", flush=True)
            with urllib.request.urlopen(ASSET_BASE + path.name, timeout=60) as response:
                payload = response.read(20 * 1024 * 1024 + 1)
            if hashlib.sha256(payload).hexdigest() != ASSET_HASHES[entry.name]:
                raise RuntimeError(f"Reference checksum mismatch: {entry.name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation does not overwrite a user's recording.
            try:
                with path.open("xb") as output:
                    output.write(payload)
            except FileExistsError:
                pass
        if download and hashlib.sha256(path.read_bytes()).hexdigest() != ASSET_HASHES[entry.name]:
            raise RuntimeError(f"Cached reference checksum mismatch: {path}")
        # The original EARS references are IEEE float WAV, not integer PCM.
        import soundfile
        with soundfile.SoundFile(str(path)) as reference:
            if reference.frames <= 0:
                raise RuntimeError(f"Empty reference WAV: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto",
                        help="Auto uses the installed native wheel; never silently falls back.")
    parser.add_argument("--server", help="Use an existing CPU/GPU server URL instead of loading a model.")
    parser.add_argument("--api-key-file", type=Path, help="Or set REALTIMETTS_API_KEY.")
    parser.add_argument("--reference-dir", type=Path, help="Original emotional reference WAV directory.")
    parser.add_argument("--cache-dir", type=Path, default=cache_root())
    parser.add_argument("--output-dir", type=Path, default=Path("qwen_emotions_output"))
    parser.add_argument("--emotions", nargs="+", choices=list(ASSET_HASHES),
                        help="Default: all eleven original showcase emotions.")
    parser.add_argument("--language", default="English")
    parser.add_argument("--text", help="Override the original spoken text for a short smoke test.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--model", type=Path, help="Optional local 0.6B Base talker GGUF.")
    parser.add_argument("--codec", type=Path, help="Optional local tokenizer GGUF.")
    parser.add_argument("--local-files-only", action="store_true", help="Disable model/reference downloads.")
    parser.add_argument("--no-play", action="store_true", help="Headless: save WAVs without importing PyAudio.")
    parser.add_argument("--list", action="store_true", help="List emotions without loading any dependencies.")
    return parser


def _check_playback() -> None:
    try:
        from ._audio_backend import pyaudio
    except ImportError as exc:
        raise RuntimeError(
            "For speaker playback install realtimetts[playback] (PortAudio is required "
            "on Linux/macOS), or use --no-play to save WAVs on a headless server."
        ) from exc
    audio = pyaudio.PyAudio()
    try:
        audio.get_default_output_device_info()
    finally:
        audio.terminate()


def _check_wav(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        frames = audio.getnframes()
        if audio.getframerate() != 24000 or audio.getnchannels() != 1 or audio.getsampwidth() != 2 or frames == 0:
            raise RuntimeError(f"Synthesis did not produce nonempty 24 kHz mono PCM: {path}")
        return frames / 24000


def _write_pcm(path: Path, chunks) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        for chunk in chunks:
            output.writeframesraw(chunk)


def _queued_pcm(engine):
    while True:
        try:
            yield engine.queue.get_nowait()
        except queue.Empty:
            return


def _native_module(device: str):
    preferred = "qwentts_cpp_cpu" if device == "cpu" else "qwentts_cpp"
    fallback = "qwentts_cpp" if device == "cpu" else "qwentts_cpp_cpu"
    try:
        return importlib.import_module(preferred)
    except ModuleNotFoundError as exc:
        if exc.name != preferred:
            raise
        return importlib.import_module(fallback)


def run_local(args, entries: list[EmotionEntry]) -> list[dict]:
    qwentts_cpp = _native_module(args.device)
    from .engines.qwen_engine import QwenEngine, QwenVoice
    from .engines.qwen_cpu_engine import QwenCpuEngine

    cpu = getattr(qwentts_cpp, "CPU_ONLY", False) is True
    device = ("cpu" if cpu else "gpu") if args.device == "auto" else args.device
    if (device == "cpu") != cpu:
        raise RuntimeError(f"Requested {device}, but the installed native wheel is {'CPU' if cpu else 'GPU'}. Use separate environments.")
    options = dict(
        model_id=MODEL_ID, quant="Q8_0", voice=None, warmup=False,
        clone_mode="speaker_only", seed=args.seed,
        voice_cache_dir=args.cache_dir / "voices",
        talker_path=args.model, codec_path=args.codec,
        local_files_only=args.local_files_only,
    )
    if cpu:
        options.update(
            cpu_threads=args.cpu_threads,
            onset_silence_profile="qwen3_tts_12hz_0_6b_base_q8_v1",
            onset_silence_recovery=True,
        )
    print(f"Loading native Qwen {device.upper()} 0.6B Base Q8_0 ...", flush=True)
    engine = (QwenCpuEngine if cpu else QwenEngine)(**options)
    results = []
    try:
        for entry in entries:
            print(f"Preparing emotional reference: {entry.name} ...", flush=True)
            entry.voice = QwenVoice(name=entry.name, ref_audio=entry.ref_audio,
                                    ref_text=entry.ref_text, language=args.language)
            engine.set_voice(entry.voice)
        engine.set_voice(entries[0].voice)
        print("Warming the loaded model ...", flush=True)
        engine.warmup()
        for entry in entries:
            engine.set_voice(entry.voice)
            text = args.text if args.text is not None else entry.speak_text
            output = args.output_dir / f"{entry.name}.wav"
            print(f"Speaking {entry.name}: {text}", flush=True)
            start = time.perf_counter()
            if args.no_play:
                if not engine.synthesize(text):
                    raise RuntimeError(f"Synthesis failed for {entry.name}: {engine.last_error}")
                _write_pcm(output, _queued_pcm(engine))
            else:
                from .text_to_stream import TextToAudioStream
                stream = TextToAudioStream(engine, on_audio_stream_start=lambda: print(
                    f"<TTFA> {time.perf_counter() - start:.3f}s", flush=True))
                stream.feed([text]).play(
                    output_wavfile=str(output), log_synthesized_text=True,
                    fast_sentence_fragment=False, force_first_fragment_after_words=9999,
                    minimum_sentence_length=25, minimum_first_fragment_length=25,
                    comma_silence_duration=0.15, sentence_silence_duration=0.3,
                    default_silence_duration=0.3,
                )
                if engine.last_error is not None:
                    raise RuntimeError(f"Synthesis failed for {entry.name}: {engine.last_error}")
            duration = _check_wav(output)
            profile = dict(engine.last_synthesis_profile)
            result = dict(emotion=entry.name, device=device, output=str(output.resolve()),
                          audio_seconds=duration, elapsed_seconds=time.perf_counter() - start,
                          last_fragment_profile=profile)
            results.append(result)
            print(f"Saved {output} ({duration:.2f}s audio)", flush=True)
    finally:
        engine.shutdown()
    return results


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward a server API key to a redirected destination.
        return None


def _request(url: str, key: str, payload=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(url, headers=headers,
                                    data=None if payload is None else json.dumps(payload).encode())
    return urllib.request.build_opener(_NoRedirect).open(request, timeout=180)


def run_server(args, entries: list[EmotionEntry]) -> list[dict]:
    base = args.server.rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise ValueError("--server must be an http:// or https:// URL")
    key = (args.api_key_file.read_text(encoding="utf-8") if args.api_key_file
           else os.environ.get("REALTIMETTS_API_KEY", "")).strip()
    with _request(base + "/v1/capabilities", key) as response:
        capabilities = json.load(response)
    model = capabilities.get("engine", {})
    if capabilities.get("features", {}).get("model_type", model.get("model_type", "base")) != "base":
        raise ValueError("The emotional-reference demo requires a Base model, not CustomVoice/VoiceDesign.")
    native_id = model.get("model_id")
    if native_id and native_id != MODEL_ID:
        raise ValueError(f"Expected the original {MODEL_ID}, server reports {native_id}.")
    device = "cpu" if model.get("cpu_only") else "gpu"
    if args.device != "auto" and args.device != device:
        raise ValueError(f"Requested {args.device}, server reports {device}.")
    results = []
    for entry in entries:
        digest = hashlib.sha256(Path(entry.ref_audio).read_bytes()).hexdigest()[:12]
        name = f"emotions-demo-{entry.name}-{digest}"
        print(f"Preparing server reference: {entry.name} ...", flush=True)
        with _request(base + "/v1/audio/voices", key, {
            "name": name, "ref_text": entry.ref_text,
            "wav_b64": base64.b64encode(Path(entry.ref_audio).read_bytes()).decode("ascii"),
        }) as response:
            json.load(response)
        output = args.output_dir / f"{entry.name}.wav"
        start = time.perf_counter()
        first_pcm_ms = None
        audio = playback = None
        try:
            if not args.no_play:
                from ._audio_backend import pyaudio
                audio = pyaudio.PyAudio()
                playback = audio.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
            with _request(base + "/v1/audio/speech", key, {
                "input": args.text if args.text is not None else entry.speak_text,
                "voice": name, "language": args.language, "clone_mode": "speaker_only",
                "seed": args.seed, "response_format": "pcm",
            }) as response:
                def chunks():
                    nonlocal first_pcm_ms
                    pending = b""
                    while True:
                        data = response.read1(8192)
                        if not data:
                            break
                        if first_pcm_ms is None:
                            first_pcm_ms = (time.perf_counter() - start) * 1000
                            print(f"<first PCM> {first_pcm_ms:.1f}ms", flush=True)
                        pending += data
                        usable = len(pending) - len(pending) % 2
                        if usable:
                            pcm, pending = pending[:usable], pending[usable:]
                            if playback is not None:
                                playback.write(pcm)
                            yield pcm
                    if pending:
                        raise RuntimeError("Server returned incomplete int16 PCM.")
                _write_pcm(output, chunks())
        finally:
            if playback is not None:
                playback.close()
            if audio is not None:
                audio.terminate()
        duration = _check_wav(output)
        results.append(dict(emotion=entry.name, device=device, output=str(output.resolve()),
                            audio_seconds=duration, first_pcm_ms=first_pcm_ms,
                            elapsed_seconds=time.perf_counter() - start))
        print(f"Saved {output} ({duration:.2f}s audio)", flush=True)
    return results


def main(argv=None, *, default_reference_dir: Optional[Path] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(ASSET_HASHES))
        return 0
    if args.text is not None and not args.text.strip():
        parser.error("--text must not be empty")
    if args.server and any((args.model, args.codec, args.cpu_threads is not None)):
        parser.error("--model, --codec and --cpu-threads configure a local engine, not an existing server")
    if args.device == "gpu" and args.cpu_threads is not None:
        parser.error("--cpu-threads requires a CPU engine")
    reference_dir = args.reference_dir or default_reference_dir or args.cache_dir / "references"
    entries = build_emotion_entries(reference_dir.resolve())
    if args.emotions:
        entries = [entry for entry in entries if entry.name in args.emotions]
    try:
        if not args.no_play:
            _check_playback()
        ensure_references(entries, download=(not args.local_files_only and
                                            args.reference_dir is None and default_reference_dir is None))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results = (run_server if args.server else run_local)(args, entries)
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("Emotional Qwen showcase completed successfully.", flush=True)
        return 0
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Qwen emotions demo failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
