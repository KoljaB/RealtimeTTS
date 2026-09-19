"""Press 1 to send a complete sentence to the CPU prefix-splice engine."""
from __future__ import annotations

import argparse
from pathlib import Path
import queue
import sys
import threading
import time
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RealtimeTTS import PrefixSpliceEngine, QwenHttpSynthesizer, TextToAudioStream
from RealtimeTTS.prefix_splice import PhoneAligner
from tools.qwen_prefix_splice import DEFAULT_TEXT


class KeyboardDemo:
    def __init__(self, engine, text, stream_factory=TextToAudioStream, clock=time.perf_counter):
        self.engine, self.text, self.clock = engine, text, clock
        self.messages = queue.Queue()
        self.worker = None
        self.pressed_at = None
        self.ttft_ms = None
        self.stream = stream_factory(engine, on_audio_stream_start=self._audio_started)

    def _audio_started(self):
        # This callback runs immediately before StreamPlayer's first device write.
        # Print in the keyboard thread, so terminal I/O cannot delay that write.
        self.ttft_ms = (self.clock()-self.pressed_at)*1000
        self.messages.put(f"TTFT: {self.ttft_ms:.1f} ms  (Taste -> erste Audioausgabe)")

    def start(self, pressed_at):
        if self.worker is not None and self.worker.is_alive():
            self.messages.put("Läuft bereits. Leertaste stoppt.")
            return False
        self.pressed_at = pressed_at
        self.ttft_ms = None
        def play():
            try:
                # All text is submitted at once: no simulated LLM/token delays.
                self.stream.feed(self.text).play()
                gaps = [x["gap_seconds"] for x in self.engine.metrics if x["event"] == "estimated_underrun"]
                if gaps:
                    self.messages.put("Geschätzte Lieferpausen: " + ", ".join(f"{g*1000:.0f} ms" for g in gaps))
                self.messages.put("Bereit: 1 = erneut sprechen")
            except Exception as error:
                self.messages.put(f"Fehler: {error}")
        self.worker = threading.Thread(target=play, name="prefix-keyboard-playback", daemon=True)
        self.worker.start()
        return True

    def stop(self):
        if self.worker is not None and self.worker.is_alive():
            self.stream.stop()
            self.worker.join(timeout=1)


def main():
    import msvcrt
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(r"D:\Projekte\miep-codex\config.toml"))
    parser.add_argument("--url", default="http://192.168.178.22:18085")
    parser.add_argument("--voice", default="mira_v5_spark_de")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--stream-initial-prefix", action="store_true")
    parser.add_argument("--immediate", action="store_true",
                        help="Experimental immediate playback; CPU delivery gaps can split words")
    args = parser.parse_args()
    key = tomllib.loads(args.config.read_text(encoding="utf-8"))["tts"]["api_key"]
    print("1/3: CPU-Qwen-Server prüfen …", flush=True)
    backend = QwenHttpSynthesizer(args.url, key, voice=args.voice)
    backend.prepare()
    print("2/3: Phonem-Aligner laden …", flush=True)
    aligner = PhoneAligner("de", str(root/".cache/phoneme_alignment"))
    print("3/3: Synthese und Alignment aufwärmen …", flush=True)
    warm = backend("Heute morgen")
    aligner.align(warm, backend.sample_rate, "Heute morgen")
    engine = PrefixSpliceEngine(backend, aligner, stream_initial_prefix=args.stream_initial_prefix,
                                continuous_playback=not args.immediate)
    demo = KeyboardDemo(engine, args.text)
    print(f"\nText: {args.text}\n", flush=True)
    print("Bereit: 1 = gesamten Text senden und sprechen | Leertaste = Stopp | Q/Esc = Ende", flush=True)
    print("TTFT misst Tastenerfassung bis direkt vor dem ersten Audio-Write, ohne Warmup.", flush=True)
    print("Sofortmodus: Lieferpausen möglich.\n" if args.immediate else
          "Flüssiger Modus: gemergten Textblock vollständig puffern, dann ausspielen.\n", flush=True)
    try:
        while True:
            while not demo.messages.empty():
                print(demo.messages.get_nowait(), flush=True)
            if msvcrt.kbhit():
                key = msvcrt.getwch().lower()
                pressed_at = time.perf_counter()
                if key in ("q", "\x1b"):
                    break
                if key == "1":
                    if demo.start(pressed_at):
                        print("Text übergeben — Synthese läuft …", flush=True)
                elif key == " ":
                    demo.stop()
            time.sleep(.002)
    except KeyboardInterrupt:
        pass
    finally:
        demo.stop()
        engine.shutdown()


if __name__ == "__main__":
    main()
