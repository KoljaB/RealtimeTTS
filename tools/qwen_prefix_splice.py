"""Offline A/B/C listening experiment; no changes to the TTS service."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata
import urllib.request
import wave
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RealtimeTTS.prefix_splice import PhoneAligner, boundary_center, ctc_spans, words

import numpy as np

PHONE_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
DEFAULT_TEXT = "Heute Morgen gehen wir gemeinsam durch den Park und genießen die frische Luft."








def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("Expected mono PCM16 WAV")
        rate = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if not len(audio) or not np.any(audio):
        raise ValueError("Empty or silent synthesis")
    return audio, rate


def write_wav(path, audio, rate):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2").tobytes())




def splice(a, b, rate, alignment_a, alignment_b, crossfade_ms=10, mode="boundary", search_ms=5):
    aw, bw = alignment_a["words"], alignment_b["words"]
    if len(aw) != 2 or len(bw) < 3 or [x["word"] for x in aw] != [x["word"] for x in bw[:2]]:
        raise ValueError("A must contain exactly B's first two words")
    for alignment, audio in ((alignment_a, a), (alignment_b, b)):
        last = 0.0
        for word in alignment["words"]:
            if not last <= word["start"] < word["end"] <= len(audio) / rate:
                raise ValueError("Invalid or unordered word boundaries")
            last = word["end"]
    if mode not in ("boundary", "gap"):
        raise ValueError("Unknown splice mode")
    overlap = round(crossfade_ms * rate / 1000)
    if overlap < 2:
        raise ValueError("Crossfade must span at least two samples")
    center_a, detail_a = boundary_center(a, rate, aw[0]["end"], aw[1]["start"], overlap, search_ms)
    center_b, detail_b = boundary_center(b, rate, bw[0]["end"], bw[1]["start"], overlap, search_ms)
    if mode == "gap":
        left, right = a[:center_a].copy(), b[center_b:].copy()
        return np.concatenate((left, right)), left, right, {
            "mode": mode, "a_cut_sample": center_a, "b_cut_sample": center_b,
            "overlap_samples": 0, "crossfade_ms": 0,
            "boundary_a": detail_a, "boundary_b": detail_b}
    # Both source centers land at the SAME output sample. This is a genuine
    # overlap around the boundary, not a fade between two trimmed word edges.
    cut_a = center_a + overlap - overlap//2
    cut_b = center_b - overlap//2
    left, right = a[:cut_a].copy(), b[cut_b:].copy()
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    joined = np.concatenate((left[:-overlap], left[-overlap:] * (1-ramp) + right[:overlap] * ramp,
                             right[overlap:]))
    return joined, left, right, {"mode": mode,
                                "boundary_a": detail_a, "boundary_b": detail_b,
                                "a_cut_sample": cut_a, "b_cut_sample": cut_b,
                                "overlap_samples": overlap, "crossfade_ms": crossfade_ms,
                                "c_transition_start_sample": len(left)-overlap,
                                "c_transition_end_sample": len(left)}


def request(base, path, key, payload=None, timeout=300):
    headers = {"Accept": "application/json" if payload is None else "audio/wav"}
    if key:
        headers["Authorization"] = "Bearer " + key
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers)
    # Never forward a bearer credential to a redirected host.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    with urllib.request.build_opener(NoRedirect).open(req, timeout=timeout) as response:
        result = response.read()
    return json.loads(result) if payload is None else result


def listen(output):
    import msvcrt
    import winsound
    print("A/1: zwei Wörter | B/2: ganzer Satz | C/3: Merge | Leertaste: Stopp | Q/Esc: Ende", flush=True)
    keys = {"a": "A.wav", "1": "A.wav", "b": "B.wav", "2": "B.wav", "c": "C.wav", "3": "C.wav"}
    for letter, number, name, description in (("d", "4", "D.wav", "10-ms-Crossfade ohne RMS-Korrektur"),
                                               ("e", "5", "C_legacy.wav", "alter 40-ms-Merge")):
        if (output / name).exists():
            keys.update({letter: name, number: name})
            print(f"{letter.upper()}/{number}: {description}", flush=True)
    try:
        while True:
            key = msvcrt.getwch().lower()
            if key in ("q", "\x1b"):
                break
            if key in keys:
                path = output / keys[key]
                print(f"Spiele {path.name}", flush=True)
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            elif key == " ":
                winsound.PlaySound(None, 0)
    finally:
        winsound.PlaySound(None, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--url", default="http://192.168.178.22:18085")
    parser.add_argument("--voice", default="mira_v5_spark_de")
    parser.add_argument("--language", default="german")
    parser.add_argument("--phoneme-language", default="de")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--config", type=Path, help="Read only [tts].api_key from an existing TOML config")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "test_outputs" / "prefix_splice")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parents[1] / ".cache" / "phoneme_alignment")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--play-only", action="store_true")
    parser.add_argument("--align-existing", action="store_true", help="Resume from A.wav/B.wav and synthesis.json")
    parser.add_argument("--remix-existing", action="store_true", help="Reuse saved A/B and alignments, no model or network")
    parser.add_argument("--crossfade-ms", type=float, default=10, help="Overlap centered at each refined word boundary")
    parser.add_argument("--boundary-search-ms", type=float, default=5, help="Maximum RMS correction per recording (0 to 5 ms)")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.play_only:
        for name in ("A", "B", "C"):
            read_wav(output / f"{name}.wav")
        listen(output)
        return
    text = unicodedata.normalize("NFKC", args.text).strip()
    word_list = words(text)
    if len(word_list) < 3 or re.search(r"\d", text):
        parser.error("Use at least three fully spelled-out words; normalize numbers first")
    prefix = " ".join(word_list[:2])
    output.mkdir(parents=True, exist_ok=True)
    if args.align_existing or args.remix_existing:
        manifest = json.loads((output / "synthesis.json").read_text(encoding="utf-8"))
        text, prefix = manifest["text"], manifest["prefix"]
        for name in ("A", "B"):
            if hashlib.sha256((output / f"{name}.wav").read_bytes()).hexdigest() != manifest["sha256"][name]:
                raise ValueError("Existing synthesis hash mismatch")
    else:
        if any((output / name).exists() for name in ("A.wav", "B.wav", "C.wav", "synthesis.json")):
            raise ValueError("Output already contains a run. Use --play-only, --align-existing, or a fresh --output")
        key = args.api_key_file.read_text(encoding="utf-8").strip() if args.api_key_file else os.environ.get("QWEN_TTS_API_KEY", "")
        if args.config:
            if key:
                parser.error("Choose either --config, --api-key-file, or QWEN_TTS_API_KEY")
            import tomllib
            key = tomllib.loads(args.config.read_text(encoding="utf-8"))["tts"]["api_key"]
        print("Prüfe CPU-Qwen-Server …", flush=True)
        capabilities = request(args.url, "/v1/capabilities", key)
        engine = capabilities["engine"]
        if engine.get("device") != "cpu" or not engine.get("cpu_only"):
            raise ValueError("Expected a verified CPU-only Qwen endpoint")
        print(json.dumps(engine, ensure_ascii=False), flush=True)
        timings = {}
        for name, content in (("A", prefix), ("B", text)):
            print(f"Synthetisiere {name}: {content}", flush=True)
            start = time.perf_counter()
            audio = request(args.url, "/v1/audio/speech", key,
                            {"input": content, "voice": args.voice, "language": args.language,
                             "response_format": "wav", "seed": 42, "do_sample": False,
                             "subtalker_do_sample": False})
            (output / f"{name}.wav").write_bytes(audio)
            read_wav(output / f"{name}.wav")
            timings[name] = time.perf_counter() - start
        manifest = {"text": text, "prefix": prefix, "engine": engine, "voice": args.voice,
                    "language": args.language, "seed": 42, "do_sample": False,
                    "subtalker_do_sample": False, "complete_wav_request_seconds": timings,
                    "sha256": {n: hashlib.sha256((output / f"{n}.wav").read_bytes()).hexdigest() for n in ("A", "B")}}
        (output / "synthesis.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.remix_existing:
        print("Lade CPU-Phonem-Aligner (erster Start lädt das Modell) …", flush=True)
        aligner = PhoneAligner(args.phoneme_language, str(args.cache_dir.resolve()))
    a, rate = read_wav(output / "A.wav")
    b, rate_b = read_wav(output / "B.wav")
    if rate != rate_b:
        raise ValueError("Sample rates differ")
    aligned = []
    for name, audio, content in (("A", a, prefix), ("B", b, text)):
        if args.remix_existing:
            result = json.loads((output / f"{name}.alignment.json").read_text(encoding="utf-8"))
            if [w["word"] for w in result["words"]] != words(content):
                raise ValueError("Saved alignment transcript mismatch")
        else:
            print(f"Phonem-Alignment {name} …", flush=True)
            result = aligner.align(audio, rate, content)
            (output / f"{name}.alignment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        aligned.append(result)
    c, left, right, cuts = splice(a, b, rate, *aligned, crossfade_ms=args.crossfade_ms, search_ms=args.boundary_search_ms)
    d, _, _, d_cuts = splice(a, b, rate, *aligned, crossfade_ms=args.crossfade_ms, search_ms=0)
    old_report = output / "report.json"
    if (output / "C.wav").exists() and old_report.exists():
        previous = json.loads(old_report.read_text(encoding="utf-8"))
        if "mode" not in previous["splice"] and not (output / "C_legacy.wav").exists():
            (output / "C_legacy.wav").write_bytes((output / "C.wav").read_bytes())
            (output / "report_legacy.json").write_bytes(old_report.read_bytes())
    for name, audio in (("C", c), ("D", d), ("A_boundary_left", left), ("B_boundary_right", right)):
        write_wav(output / f"{name}.wav", audio, rate)
    report = {**manifest, "splice": cuts, "no_rms_splice": d_cuts, "sample_rate": rate,
              "output_sha256": {n: hashlib.sha256((output / f"{n}.wav").read_bytes()).hexdigest() for n in ("C", "D")},
              "duration_seconds": {"A": len(a)/rate, "B": len(b)/rate, "C": len(c)/rate, "D": len(d)/rate},
              "limitations": ["Offline listening experiment, not a TTFA benchmark.",
                              "No punctuation in A; native generation still ends the utterance.",
                              "Fixed seed and greedy sampling do not guarantee prefix-invariant prosody.",
                              "Phoneme CTC boundaries are estimates on a 20 ms grid.",
                              "CTC blank intervals are not silence; local RMS only refines an estimated word boundary.",
                              "Crossfade is centered on the refined boundary; compare C and D by listening."]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Fertig: {output / 'C.wav'}", flush=True)
    if not args.prepare_only:
        listen(output)


if __name__ == "__main__":
    main()
