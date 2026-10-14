"""Explain streaming completion failures without changing the completion policy."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from RealtimeTTS.alignment import streaming
from RealtimeTTS.alignment.streaming import StreamingTextAligner, StreamingAlignmentSession

def evidence(session):
    policy=session.owner.completion_policy
    measured={c.index:c for c in session.characters}
    rows=[]
    for word in re.finditer(r"\S+",session.tracker.text):
        chars=[measured.get(i) for i in range(word.start(),word.end())]
        spoken=[c for c in chars if c is not None and any(x.isalnum() for x in c.character)]
        scores=[c.score for c in spoken]
        support=sum(s is not None and s>=policy.minimum_token_probability for s in scores)
        rows.append({"word":word.group(),"missing":sum(c is None for c in chars),
                     "supported_fraction":support/len(scores) if scores else 0,
                     "scores":scores,"end":max((c.end_time for c in spoken),default=None)})
    chars=[c for c in session.characters if any(x.isalnum() for x in c.character)]
    total=sum(c.score is not None and c.score>=policy.minimum_token_probability for c in chars)
    overall=total/len(chars) if chars else 0
    reasons=[]
    if not session.complete:reasons.append("coverage")
    if overall<policy.minimum_supported_fraction:reasons.append("overall_support")
    if any(len(r["word"])>=policy.word_support_minimum_length and r["supported_fraction"]<policy.minimum_word_supported_fraction for r in rows):reasons.append("word_support")
    if rows and rows[-1]["supported_fraction"]<policy.minimum_suffix_supported_fraction:reasons.append("suffix_support")
    for name,margin in (("mean_margin",session.completion_mean_log_margin),("suffix_margin",session.completion_final_log_margin)):
        if margin is not None and (not np.isfinite(margin) or margin<policy.minimum_mean_log_margin):reasons.append(name)
    return {"audio_seconds":session.analyzed_seconds,"complete":session.complete,
            "verified":session.completion_verified,"overall_support":overall,
            "reasons":reasons,"words":rows,
            "mean_margin":session.completion_mean_log_margin,
            "suffix_margin":session.completion_final_log_margin,
            "boundary":session.completion_boundary_observed}

def probe(aligner,case,audio,rate,variant,full_window=False):
    trace=[]
    original_verify=StreamingAlignmentSession._verify_completion
    original_anchor=streaming.window_anchor
    def verify(session,final):
        original_verify(session,final)
        trace.append({**evidence(session),"final":final})
    StreamingAlignmentSession._verify_completion=verify
    if full_window:streaming.window_anchor=lambda *a,**kw:(0,0,0.0)
    events=[]
    session=aligner.start(case["text"],rate,language=case.get("language","english"),callback=events.append)
    hop=round(session.interval_seconds*rate)
    padded=np.pad(audio,(0,round(.64*rate)))
    padded=np.pad(padded,(0,(-len(padded))%hop))
    try:
        for end in range(hop,len(padded)+1,hop):
            session.feed(padded[end-hop:end])
            deadline=time.monotonic()+30
            while session.analyzed_seconds+1e-6<end/rate and not session.done.is_set():
                if time.monotonic()>deadline:raise TimeoutError("worker did not consume prefix")
                session.done.wait(.002)
            if session.done.is_set():break
        session.finish()
        if not session.done.wait(30):raise TimeoutError("worker did not finish")
        completion=next((e for e in events if e["type"]=="alignment_complete"),None)
        result={"variant":variant,"full_window":full_window,"error":session.error,
                "stop":session.stop_requested.is_set(),"finish_reason":session.finish_reason,
                "completion":completion,"evidence":evidence(session),"trace":trace,
                "words":[asdict(w) for w in session.words]}
        return result
    finally:
        session.finish(cancelled=True)
        session.done.wait(30)
        StreamingAlignmentSession._verify_completion=original_verify
        streaming.window_anchor=original_anchor

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--multilingual-model",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--ids",nargs="*")
    parser.add_argument("--compare-full-window",action="store_true")
    parser.add_argument("--extra-speech",action="store_true")
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    cases=json.loads(args.manifest.read_text(encoding="utf-8"))["cases"]
    if args.ids:cases=[c for c in cases if c["id"] in args.ids]
    aligner=StreamingTextAligner(model_dir=args.model,language="english",multilingual_model=args.multilingual_model,window_seconds=30)
    report={"policy":asdict(aligner.completion_policy),"cases":[],"source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("RealtimeTTS/alignment/streaming.py","RealtimeTTS/alignment/evidence.py","RealtimeTTS/alignment/windows.py","RealtimeTTS/alignment/onnx_parakeet.py")}}
    try:
        for case in cases:
            audio_path=Path(case["audio"])
            if not audio_path.is_absolute():audio_path=args.manifest.parent/audio_path
            audio,rate=sf.read(audio_path,dtype="float32")
            backend=aligner.backend if case.get("language","english")=="english" else aligner.multilingual_backend
            if hasattr(backend,"recognize"):
                recognized=backend.recognize(audio,rate)
                decoding="".join(recognized.tokens)
            else:
                emissions=backend.emissions(audio,rate)
                ids=emissions.argmax(axis=1);inverse={v:k for k,v in backend.tokens.items()}
                decoding="".join(inverse.get(int(v),"?") for i,v in enumerate(ids) if v!=backend.blank_id and (i==0 or ids[i-1]!=v))
            runs=[probe(aligner,case,audio,rate,"whole")]
            if args.compare_full_window:runs.append(probe(aligner,case,audio,rate,"whole",True))
            if args.extra_speech:runs.append(probe(aligner,case,np.r_[audio,np.zeros(round(.3*rate),dtype=np.float32),audio],rate,"extra_speech"))
            item={"id":case["id"],"text":case["text"],"language":case.get("language","english"),"duration":len(audio)/rate,"decoded":decoding,"runs":runs}
            report["cases"].append(item)
            for run in runs:
                ev=run["evidence"]
                print(json.dumps({"id":case["id"],"variant":run["variant"],"full":run["full_window"],"stop":run["stop"],"reasons":ev["reasons"],"weak":[r["word"] for r in ev["words"] if r["missing"] or r["supported_fraction"]<.5]}),flush=True)
    finally:aligner.close()
    with args.output.open("x",encoding="utf-8") as handle:json.dump(report,handle,ensure_ascii=False,indent=2)
    print(json.dumps({"output":str(args.output),"cases":len(report["cases"])}))

if __name__=="__main__":main()
