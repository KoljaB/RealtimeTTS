"""Checks against human labels, not detector-created labels."""
from pathlib import Path
import importlib.util
import json
import sys
import wave
import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "tts_quality_check.py"
spec = importlib.util.spec_from_file_location("tts_quality_check", SOURCE)
qc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qc
spec.loader.exec_module(qc)
MANIFEST = Path(r"D:\Temp\Codex\tts-acoustic-detection-20261013\corpus-manifest.json")
GOLD = (json.loads(MANIFEST.read_text(encoding="utf-8-sig"))["records"] if MANIFEST.exists() else [])
GOLD = [r for r in GOLD if r["label"] in {"failure", "clean"}]

@pytest.mark.skipif(not GOLD, reason="Private recorded corpus is not installed")
@pytest.mark.parametrize("rec", GOLD, ids=[Path(r["path"]).name for r in GOLD])
def test_human_reviewed_recordings(rec):
    result = qc.check_wav(Path(rec["path"]))
    assert result["audio"]["sha256"] == rec["sha256"]
    assert result["strange_noise_suspected"] == (rec["label"] == "failure")
    if Path(rec["path"]).name == "tutor-spelling-2.wav":
        assert result["long_silence_suspected"]
        assert result["status"] == "silence_suspected"

@pytest.mark.skipif(not GOLD, reason="Private recorded corpus is not installed")
@pytest.mark.parametrize("gain_db", [-6, 6])
def test_gain_changes_do_not_change_gold_noise_labels(gain_db):
    for rec in GOLD:
        x, sr, _ = qc.read_wav(Path(rec["path"]))
        x = np.clip(x * 10 ** (gain_db / 20), -1, 32767 / 32768)
        result = qc.analyze(x, sr)
        assert result["strange_noise_suspected"] == (rec["label"] == "failure"), rec["path"]

@pytest.mark.skipif(not GOLD, reason="Private recorded corpus is not installed")
def test_normal_speech_resumes_after_long_pause():
    rec = next(r for r in GOLD if Path(r["path"]).name == "mixed-refcompare-0-0.wav")
    x, sr, _ = qc.read_wav(Path(rec["path"]))
    result = qc.analyze(np.concatenate([x, np.zeros((sr * 3, x.shape[1])), x]), sr)
    assert result["long_silence_suspected"]
    assert not result["strange_noise_suspected"]

@pytest.mark.parametrize("seconds,flag", [(1.0, False), (1.005, True), (2.0, True)])
def test_strict_one_second_silence_boundary(seconds, flag):
    result = qc.analyze(np.zeros(round(16000 * seconds)), 16000)
    assert result["long_silence_suspected"] is flag
    assert not result["strange_noise_suspected"]

@pytest.mark.parametrize("audio", [np.empty(0), np.array([np.nan]), np.array([np.inf])])
def test_invalid_audio_is_never_clean(audio):
    with pytest.raises(ValueError):
        qc.analyze(audio, 16000)

def test_malformed_wav_is_rejected(tmp_path):
    path = tmp_path / "bad.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(160, dtype="<i2").tobytes())
    with path.open("ab") as f:
        f.write(b"extra_pcm")
    with pytest.raises(ValueError, match="RIFF"):
        qc.check_wav(path)

def test_missing_word_times_do_not_invent_an_asr_tail():
    result = qc.analyze(np.zeros(32000), 16000, asr={"response": {"text": "hola", "segments": []}})
    assert result["asr_status"] == "word_timestamps_unavailable"
    assert not result["strange_noise_suspected"]

def test_asr_tail_exposes_only_audible_intervals():
    sr = 16000
    x = np.zeros(sr * 5)
    t = np.arange(sr // 4) / sr
    x[sr * 3:sr * 3 + len(t)] = .1 * np.sin(2 * np.pi * 180 * t)
    result = qc.analyze(x, sr, asr={"response": {"words": [{"word": "hola", "start": .1, "end": .5}]}})
    event = next(e for e in result["events"] if e.get("reason") == "audible_audio_after_last_asr_word")
    assert event["start_s"] >= 3.0
    assert event["end_s"] <= 3.26
    assert event["audible_intervals_s"] == [[event["start_s"], event["end_s"]]]

@pytest.mark.skipif(not GOLD, reason="Private recorded corpus is not installed")
@pytest.mark.parametrize("rec", GOLD, ids=[Path(r["path"]).name for r in GOLD])
def test_cached_faster_whisper_timestamps_match_human_labels(rec):
    path = Path(r"D:\Temp\User\tts-strange-noise-detector-20261013\faster-whisper") / (rec["id"] + ".json")
    if not path.exists():
        pytest.skip("Offline large-v2 result is not installed")
    asr = json.loads(path.read_text(encoding="utf-8"))
    result = qc.check_wav(Path(rec["path"]), asr)
    assert result["asr_status"] == "word_timestamps_available"
    tail = [e for e in result["events"] if e.get("reason") == "audible_audio_after_last_asr_word"]
    assert bool(tail) == (rec["label"] == "failure")
    assert result["strange_noise_suspected"] == (rec["label"] == "failure")


LEADING_MANIFEST = Path(r"D:\Temp\User\tts-leading-noise-20261014\leading-human-manifest.json")
LEADING_GOLD = (json.loads(LEADING_MANIFEST.read_text(encoding="utf-8"))["records"]
                if LEADING_MANIFEST.exists() else [])
LEADING_REASON = "isolated_weak_tonal_audio_before_speech"


def leading_events(result):
    return [event for event in result["events"] if event.get("reason") == LEADING_REASON]


@pytest.mark.skipif(not LEADING_GOLD, reason="Private leading-noise recordings are not installed")
@pytest.mark.parametrize("rec", LEADING_GOLD, ids=[rec["id"] for rec in LEADING_GOLD])
def test_human_confirmed_leading_noise_has_its_own_event(rec):
    asr = json.loads(Path(rec["asr_path"]).read_text(encoding="utf-8"))
    result = qc.check_wav(Path(rec["path"]), asr)
    assert result["audio"]["sha256"] == rec["sha256"]
    event, = leading_events(result)
    assert event["start_s"] <= .02
    assert event["end_s"] <= .08
    assert event["following_audio_start_s"] >= .4
    assert event["first_asr_word_end_s"] > event["following_audio_start_s"]
    assert event["asr_guard_requires_full_result"]
    assert event["evidence_available_s"] == result["audio"]["duration_s"]
    if rec["id"] == "mas_despacio_seed43":
        tail, = [e for e in result["events"] if e.get("reason") == "audible_audio_after_last_asr_word"]
        assert tail["start_s"] >= 2.28
        assert event["end_s"] < tail["start_s"]


@pytest.mark.skipif(not LEADING_GOLD, reason="Private leading-noise recordings are not installed")
@pytest.mark.parametrize("gain_db", [-6, 0, 6])
def test_leading_rule_does_not_depend_on_the_bad_tail_or_asr(gain_db):
    for rec in LEADING_GOLD:
        x, sr, _ = qc.read_wav(Path(rec["path"]))
        # The first second excludes the separately confirmed late noise.
        prefix = np.clip(x[:sr] * 10 ** (gain_db / 20), -1, 32767 / 32768)
        result = qc.analyze(prefix, sr)
        event, = leading_events(result)
        assert result["strange_noise_suspected"]
        assert event["end_s"] <= .08
        assert event["evidence_available_s"] <= .8
        assert not event["asr_guard_requires_full_result"]
        assert not any(e.get("reason") == "audible_audio_after_last_asr_word" for e in result["events"])


def onset_fixture(lead_s=.04, gap_s=.36, lead_gain=.008, lead_kind="tone"):
    sr = 16000
    n = round(sr * lead_s)
    if lead_kind == "tone":
        lead = lead_gain * np.sin(2 * np.pi * 150 * np.arange(n) / sr)
    else:
        lead = lead_gain * np.random.default_rng(42).normal(size=n)
    voice = .25 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    return np.r_[lead, np.zeros(round(sr * gap_s)), voice], sr


@pytest.mark.parametrize("lead_s,gap_s,lead_gain,lead_kind", [
    (.16, .36, .008, "tone"),  # a sustained quiet syllable
    (.04, .04, .008, "tone"),  # normal consonant-to-vowel spacing
    (.04, .36, .08, "tone"),   # ordinary speech-level initial sound
    (.04, .36, .008, "noise"), # breath/fricative, not low-tone noise
])
def test_normal_onset_shapes_do_not_trigger_leading_noise(lead_s, gap_s, lead_gain, lead_kind):
    x, sr = onset_fixture(lead_s, gap_s, lead_gain, lead_kind)
    assert not leading_events(qc.analyze(x, sr))


def test_asr_recognized_quiet_first_word_is_not_leading_noise():
    x, sr = onset_fixture()
    assert leading_events(qc.analyze(x, sr))
    asr = {"response": {"words": [
        {"word": "el", "start": 0, "end": .08},
        {"word": "paraguas", "start": .4, "end": 1.4}]}}
    assert not leading_events(qc.analyze(x, sr, asr=asr))


def test_silence_only_before_speech_is_not_leading_noise():
    x, sr = onset_fixture()
    x[:round(sr * .04)] = 0
    assert not leading_events(qc.analyze(x, sr))


def test_leading_evidence_requires_a_complete_following_window():
    x, sr = onset_fixture()
    assert not leading_events(qc.analyze(x[:round(sr * .5)], sr))
    event, = leading_events(qc.analyze(x[:round(sr * .52)], sr))
    assert event["evidence_available_s"] == pytest.approx(.52)
    assert event["end_s"] == pytest.approx(.04)


def test_antiphase_stereo_does_not_cancel_leading_noise():
    x, sr = onset_fixture()
    stereo = np.column_stack([x, -x])
    assert leading_events(qc.analyze(stereo, sr))


def test_asr_hash_mismatch_is_rejected():
    if not GOLD:
        pytest.skip("Private recorded corpus is not installed")
    with pytest.raises(ValueError, match="SHA256"):
        qc.check_wav(Path(GOLD[0]["path"]), {"sha256": "wrong"})
