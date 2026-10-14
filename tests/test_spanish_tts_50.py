from pathlib import Path
import importlib.util
import copy
import json
import sys
import pytest
TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("spanish_benchmark", TOOLS / "benchmark_spanish_tts_50.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

def test_exact_fifty_diverse_cases():
    corpus = json.loads(bench.CORPUS.read_text(encoding="utf-8-sig"))
    entries = bench.validate_corpus(corpus)
    assert len({e["text"] for e in entries}) == 50
    assert sum(1 <= len(e["text"].split()) <= 3 for e in entries) == 40
    assert corpus["seed"] == 42

def test_exact_hundred_appends_original_fifty_cases():
    original = json.loads(bench.CORPUS.read_text(encoding="utf-8-sig"))
    extended_path = TOOLS / "data" / "spanish_tts_100.json"
    extended = json.loads(extended_path.read_text(encoding="utf-8-sig"))
    entries = bench.validate_corpus(extended)
    assert len(entries) == 100
    assert sum(entry["class"] == "short" for entry in entries) == 90
    assert sum(entry["class"] == "longer" for entry in entries) == 10
    assert entries[:50] == original["entries"]
    assert len({bench.normalized_text(entry["text"]) for entry in entries}) == 100
    assert extended["counts"] == {"total": 100, "short_1_to_3_words": 90, "longer": 10}


def test_bad_corpus_is_rejected():
    corpus = json.loads(bench.CORPUS.read_text(encoding="utf-8-sig"))
    corpus["entries"][0]["text"] = "uno dos tres cuatro"
    with pytest.raises(ValueError):
        bench.validate_corpus(corpus)

def row(noise=False, pause=False, asr=True, synthesis=True):
    return {"tts":{"status":"complete" if synthesis else "error"},
        "asr_status":"complete" if asr else "error",
        "check":{"strange_noise_suspected":noise,"long_silence_suspected":pause,
                 "asr_status":"word_timestamps_available"} if synthesis else None}

def test_auth_key_expansion_and_missing_key(monkeypatch):
    monkeypatch.delenv("TTS_BENCH_TEST_TOKEN", raising=False)
    placeholder = chr(36) + "{TTS_BENCH_TEST_TOKEN}"
    assert bench.resolve_key({"api_key":placeholder}, {"TTS_BENCH_TEST_TOKEN":"test-value"}) == "test-value"
    with pytest.raises(ValueError):
        bench.resolve_key({"api_key":placeholder}, {})

def test_counts_separate_noise_silence_and_errors():
    counts = bench.totals([row(noise=True,pause=True),row(pause=True),row(),row(asr=False),row(synthesis=False)])
    assert counts["noise_suspected"] == 1
    assert counts["long_silence"] == 2
    assert counts["silence_only"] == 1
    assert counts["no_anomaly_found"] == 1
    assert counts["unclassified"] == 2
    assert counts["synthesis_errors"] == 1
    assert counts["human_confirmed_hallucinations"] is None


def test_exact_hundred_english_cases():
    corpus = json.loads((TOOLS / "data" / "english_tts_100.json").read_text(encoding="utf-8"))
    entries = bench.validate_corpus(corpus)
    assert corpus["language"] == "en"
    assert len(entries) == 100
    assert sum(entry["class"] == "short" for entry in entries) == 90
    assert sum(entry["class"] == "longer" for entry in entries) == 10
    assert all(1 <= entry["word_count"] <= 3 for entry in entries if entry["class"] == "short")


def test_cli_rejects_corpus_language_mismatch(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--language", "en", "--corpus", str(bench.CORPUS), "--validate-corpus"])
    with pytest.raises(ValueError, match="Corpus language"):
        bench.main()


def test_cli_validates_english_without_inference(monkeypatch, capsys):
    corpus_path = TOOLS / "data" / "english_tts_100.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--language", "en", "--corpus", str(corpus_path), "--validate-corpus"])
    assert bench.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert (report["total"], report["short"], report["longer"], report["language"]) == (100, 90, 10, "en")
