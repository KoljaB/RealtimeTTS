"""Live prefix-splice demo, with actual queue timing and source WAV evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import tomllib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RealtimeTTS import PrefixSpliceEngine, QwenHttpSynthesizer, TextToAudioStream
from RealtimeTTS.prefix_splice import PhoneAligner
from tools.qwen_prefix_splice import DEFAULT_TEXT, write_wav


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url", default="http://192.168.178.22:18085")
    parser.add_argument("--voice", default="mira_v5_spark_de")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--word-delay", type=float, default=.08)
    parser.add_argument("--muted", action="store_true")
    parser.add_argument("--continuous-playback", action="store_true")
    parser.add_argument("--no-streaming", action="store_true", help="Compare the previous complete-WAV path")
    parser.add_argument("--stream-initial-prefix", action="store_true", help="Also try partial alignment on the initial two-word stream")
    parser.add_argument("--output", type=Path, default=Path("test_outputs/prefix_stream"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    key = tomllib.loads(args.config.read_text(encoding="utf-8"))["tts"]["api_key"]
    backend = QwenHttpSynthesizer(args.url, key, args.voice)
    print("Prüfe CPU-Qwen …", flush=True)
    backend.prepare()
    print("Lade Phonem-Aligner vor Playback …", flush=True)
    aligner = PhoneAligner("de", str(Path(__file__).resolve().parents[1]/".cache/phoneme_alignment"))
    print("Wärme Synthese und Alignment auf …", flush=True)
    warm = backend("Heute morgen")
    aligner.align(warm, backend.sample_rate, "Heute morgen")
    sources = []
    def synthesize(text):
        print(f"Synthese {len(sources)+1}: {text}", flush=True)
        result = backend(text)
        path = args.output/f"source_{len(sources)+1}.wav"
        write_wav(path, result, backend.sample_rate)
        sources.append({"text": text, "path": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return result
    synthesize.bind_cancel_event = backend.bind_cancel_event
    synthesize.cancel = backend.cancel
    def stream_synthesis(text):
        print(f"PCM-Stream {len(sources)+1}: {text}", flush=True)
        path = args.output/f"source_{len(sources)+1}.wav"
        entry = {"text": text, "path": path.name, "complete": False}
        sources.append(entry)
        chunks = []
        try:
            for chunk in backend.stream(text):
                chunks.append(chunk)
                yield chunk
            entry["complete"] = True
        finally:
            if chunks:
                write_wav(path, np.concatenate(chunks), backend.sample_rate)
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    synthesize.stream = stream_synthesis
    alignments = []
    class RecordingAligner:
        def align_prefix(self, audio, rate, text, boundary_count):
            result = aligner.align_prefix(audio, rate, text, boundary_count)
            path = args.output/f"partial_alignment_{len(alignments)+1}.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            alignments.append(path.name)
            return result
        def align(self, audio, rate, text):
            result = aligner.align(audio, rate, text)
            path = args.output/f"alignment_{len(alignments)+1}.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            alignments.append(path.name)
            return result
    engine = PrefixSpliceEngine(synthesize, RecordingAligner(), streaming=not args.no_streaming,
                               stream_initial_prefix=args.stream_initial_prefix,
                               continuous_playback=args.continuous_playback)
    def source():
        for word in args.text.split():
            time.sleep(args.word_delay)
            yield word+" "
    played = []
    first = None
    started = time.perf_counter()
    def received(data):
        nonlocal first
        now = time.perf_counter()-started
        if first is None:
            first = now
            print(f"Erster Playback-Callback: {now:.3f} s", flush=True)
        played.append({"seconds": now, "samples": len(data)//2})
    stream = TextToAudioStream(engine, muted=args.muted)
    error = None
    try:
        stream.feed(source()).play(output_wavfile=str(args.output/"stream.wav"), on_audio_chunk=received)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        report = {"sources": sources, "events": engine.metrics, "playback_callbacks": played,
                  "alignments": alignments,
                  "backend": backend.capabilities["engine"], "word_delay": args.word_delay,
                  "muted": args.muted, "error": error,
                  "note": "Queue commit and software playback callbacks are not physical audible onset."}
        (args.output/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        engine.shutdown()
    print(f"Fertig: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
