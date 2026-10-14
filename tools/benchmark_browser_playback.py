"""Serve a real-WebAudio check using Studio's exact scheduler and native WebSocket server.

The API key is read only from an environment variable and is served only on
loopback for this local test. No microphone or other applications are recorded.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]
WORKLET="""
class Capture extends AudioWorkletProcessor {
 constructor(){super();this.buffer=new Float32Array(4096);this.used=0;this.start=0;}
 flush(){
  if(!this.used)return;
  const samples=this.used===this.buffer.length?this.buffer:this.buffer.slice(0,this.used);
  this.port.postMessage({start:this.start,samples},[samples.buffer]);
  this.buffer=new Float32Array(4096);this.used=0;
 }
 process(inputs,outputs){
  if(this.used&&currentFrame!==this.start+this.used)this.flush();
  const input=inputs[0]?.[0],out=outputs[0]?.[0];if(out&&input)out.set(input);
  const n=out?.length||128;
  for(let i=0;i<n;i++){
   if(this.used===0)this.start=currentFrame+i;
   this.buffer[this.used++]=input?.[i]||0;
   if(this.used===this.buffer.length){
    this.flush();
   }
  }
  return true;
 }
}
registerProcessor('capture',Capture);
"""
CONTROLLED=r"""

async function feedControlled(item,r,ctx,events,started){
 const response=await fetch('/fixture.pcm');
 if(!response.ok)throw Error('Controlled PCM fixture unavailable');
 const original=new Uint8Array(await response.arrayBuffer()),frames=original.length/2;
 if(frames!==item.fixture_frames||ctx.sampleRate!==24000)throw Error('Fixture or context rate changed');
 const mode=item.controlled,repeat=mode!=='original',gapFrames=7200;
 const pcm=new Uint8Array(original.length+(repeat?gapFrames*2+original.length:0));
 pcm.set(original);if(repeat)pcm.set(original,original.length+gapFrames*2);
 const identity={session_id:'controlled-session',request_id:item.id,fragment_index:0};
 const dispatch=event=>{
  const accepted=handlePlaybackGuardEvent(r,event);
  events.push({...event,accepted,wall_ms:performance.now()-started,context_time:ctx.currentTime});
  return accepted;
 };
 const cutoff={...identity,type:'alignment_complete',completion_verified:true,stop_requested:true,
  end_time:frames/24000,trim_end_time:frames/24000+.25};
 $('bufferMs').value='80';
 r.controlled={mode,fixture_frames:frames,fixture_pcm_sha256:item.fixture_pcm_sha256,
  fixture_wav_sha256:item.fixture_wav_sha256,fixture_source:item.fixture_source,
  playback_lead_ms:80,intended_ranges:[[0,frames]],surplus_ranges:repeat?[[frames+gapFrames,pcm.length/2]]:[]};
 const deadline=performance.now()+30000;
 dispatch({...identity,type:'fragment_start',sample_rate_hz:24000});
 receivePCM(r,pcm.buffer);
 if(mode==='queued-cutoff'||mode==='following-fragment')dispatch(cutoff);
 if(mode==='stale-cutoff')dispatch({...cutoff,session_id:'stale-session'});
 if(mode==='late-cutoff'){
  const target=r.playbackSegments[0].when+(frames+gapFrames)/24000+.18;
  while(ctx.currentTime<target&&performance.now()<deadline)await sleep(5);
  if(ctx.currentTime<target)throw Error('AudioContext stalled before late cutoff');
  dispatch(cutoff);
 }
 dispatch({...identity,type:'fragment_end'});
 if(mode.startsWith('following-')){
  const next={...identity,request_id:item.id+'-next',fragment_index:1};
  dispatch({...next,type:'fragment_start',sample_rate_hz:24000});
  r.controlled.intended_ranges.push([r.samples,r.samples+frames]);
  receivePCM(r,original.buffer);
  dispatch({...next,type:'fragment_end'});
  if(mode==='following-late-metadata'){await sleep(40);dispatch(cutoff);}
 }
 while(r.pending&&performance.now()<deadline)await sleep(10);
 if(r.pending)throw Error('Controlled playback deadline exceeded');
 dispatch({...identity,type:'done'});
}
function measureControlled(r,rendered,startFrame,sampleRate,received){
 if(!r.controlled)return null;
 if(sampleRate!==24000||!Number.isInteger(startFrame))throw Error('Cannot map captured frames');
 const reference=new DataView(received.buffer,received.byteOffset,received.byteLength),threshold=256;
 const inside=(sample,ranges)=>ranges.some(([a,b])=>sample>=a&&sample<b);
 const intendedOutput=[];
 for(const segment of r.playbackSegments)for(const [a,b] of r.controlled.intended_ranges){
  const lo=Math.max(a,segment.sampleStart),hi=Math.min(b,segment.sampleStart+segment.sampleCount);
  if(lo<hi){const origin=Math.round(segment.when*sampleRate)-startFrame;
   intendedOutput.push([origin+lo-segment.sampleStart,origin+hi-segment.sampleStart]);}
 }
 let legitimate=0,missing=0,changed=0,surplusSignal=0,maxError=0,lastSurplus=-1,firstSurplus=-1;
 let totalReference=0,squaredError=0;
 for(const segment of r.playbackSegments){
  const outputStart=Math.round(segment.when*sampleRate)-startFrame;
  for(let i=0;i<segment.sampleCount;i++){
   const sample=segment.sampleStart+i,at=outputStart+i,expected=reference.getInt16(sample*2,true);
   const actual=at>=0&&at<rendered.length?Math.round(rendered[at]*32768):0;
   if(inside(sample,r.controlled.intended_ranges)){
    const error=Math.abs(actual-expected);maxError=Math.max(maxError,error);
    totalReference+=expected*expected;squaredError+=(actual-expected)**2;
    if(Math.abs(expected)>=threshold){
     legitimate++;if(Math.abs(actual)<threshold)missing++;if(error>2)changed++;
    }
   }
   if(inside(sample,r.controlled.surplus_ranges)&&!inside(at,intendedOutput)&&Math.abs(actual)>=threshold){
    surplusSignal++;if(firstSurplus<0)firstSurplus=sample;lastSurplus=sample;
   }
  }
 }
 const mode=r.controlled.mode,shouldTrim=['queued-cutoff','following-fragment','following-late-metadata'].includes(mode);
 const surplusStart=r.controlled.surplus_ranges[0]?.[0]??0;
 const lastSurplusMs=lastSurplus<0?0:(lastSurplus+1-surplusStart)/sampleRate*1000;
 const checks={legitimate_signal_present:legitimate>0,no_lost_legitimate_signal:missing===0,
  legitimate_samples_preserved:changed===0,all_intended_pcm_preserved:maxError<=2};
 if(shouldTrim)checks.no_surplus_signal=surplusSignal===0;
 if(['repeat-control','stale-cutoff','late-cutoff'].includes(mode))checks.surplus_control_detected=surplusSignal>0;
 let nextFragmentGapMs=null;
 if(mode.startsWith('following-')){
  const nextStart=r.controlled.intended_ranges[1][0];
  const segment=r.playbackSegments.find(s=>s.sampleStart===nextStart);
  const first=r.playbackSegments[0];
  nextFragmentGapMs=(segment.when-(first.when+r.controlled.fixture_frames/sampleRate+.25))*1000;
  checks.no_wait_for_trimmed_audio=Math.abs(nextFragmentGapMs)<=1;
 }
 return {...r.controlled,signal_threshold_pcm:threshold,legitimate_signal_samples:legitimate,
  lost_legitimate_signal_samples:missing,changed_legitimate_signal_samples:changed,
  peak_legitimate_error_pcm:maxError,relative_squared_error:totalReference?squaredError/totalReference:null,
  surplus_signal_samples:surplusSignal,surplus_signal_ms:surplusSignal/sampleRate*1000,
  last_surplus_after_boundary_ms:lastSurplusMs,first_surplus_input_sample:firstSurplus,
  next_fragment_gap_after_cutoff_ms:nextFragmentGapMs,checks,passed:Object.values(checks).every(Boolean)};
}
"""
HARNESS=r"""
const controls=new Map(),recorded=[];let run=null;
const $=id=>{if(!controls.has(id))controls.set(id,{value:'0',textContent:''});return controls.get(id);};
function alignmentPlaybackLeadSeconds(){return 0;}
function drawWave(){}function maybeFinalize(){}
function stop(){throw Error('Unexpected buffer overflow');}
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
function encodePCM(samples){
 const data=new Uint8Array(samples.length*2),view=new DataView(data.buffer);
 for(let i=0;i<samples.length;i++)view.setInt16(i*2,Math.max(-32768,Math.min(32767,Math.round(samples[i]*32768))),true);
 let s='';for(let i=0;i<data.length;i+=32768)s+=String.fromCharCode(...data.subarray(i,i+32768));
 return btoa(s);
}
async function checkCase(item,index){
 controls.clear();
 const ctx=new AudioContext({sampleRate:24000,latencyHint:'interactive'});
 await ctx.resume();await ctx.audioWorklet.addModule('/worklet.js');
 if(ctx.state!=='running')throw Error('AudioContext did not start');
 const tap=new AudioWorkletNode(ctx,'capture',{numberOfInputs:1,numberOfOutputs:1,outputChannelCount:[1]});
 tap.connect(ctx.destination);const blocks=[];let firstRenderedClock=null;
 tap.port.onmessage=e=>{
  const block=e.data;blocks.push(block);
  if(firstRenderedClock===null){
   const signal=block.samples.findIndex(x=>Math.abs(x)>=256/32768);
   if(signal>=0)firstRenderedClock={sample_frame:block.start+signal,
    observed_performance_ms:performance.now(),output_timestamp:ctx.getOutputTimestamp?.()};
  }
 };
 // Studio primes its shared context before requests; settle the device clock likewise.
 const prime=ctx.createBufferSource();prime.buffer=ctx.createBuffer(1,2400,24000);prime.connect(tap);
 await new Promise(resolve=>{prime.onended=resolve;prime.start();});prime.disconnect();
 blocks.length=0;firstRenderedClock=null;
 const proxy=new Proxy(ctx,{get(target,prop){if(prop==='destination')return tap;const v=Reflect.get(target,prop,target);return typeof v==='function'?v.bind(target):v;}});
 const started=performance.now();
 const r={ctx:proxy,samples:0,next:0,odd:null,pending:0,sources:new Set(),preReady:[],
  audioReady:true,started,audible:false,underruns:0,gapMs:0,playbackSegments:[],alignment:{enabled:false}};
 run=r;recorded.length=0;const events=[];
 let failed=null;
 if(item.controlled){
  try{await feedControlled(item,r,ctx,events,started);}catch(error){failed={message:String(error)};}
 }else{
 const socket=new WebSocket(CONFIG.base.replace(/^http/,'ws')+'/v1/audio/speech-stream',
   ['qwen-studio','bearer.'+btoa(CONFIG.key).replace(/=/g,'').replace(/\+/g,'-').replace(/\//g,'_')]);
 socket.binaryType='arraybuffer';
 let ended=false;
 await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=()=>reject(Error('WebSocket open failed'));});
 socket.onmessage=e=>{
  if(typeof e.data!=='string'){receivePCM(r,e.data);return;}
  const event=JSON.parse(e.data);
  events.push({...event,wall_ms:performance.now()-started,context_time:ctx.currentTime});
  handlePlaybackGuardEvent(r,event);
  if(event.type==='error')failed=event;
  if(event.type==='done'||event.type==='cancelled')ended=true;
 };
 socket.onclose=()=>{ended=true;};
 socket.send(JSON.stringify({type:'config',voice:item.voice,language:item.language,
  seed:item.seed??20261009+index,max_new_tokens:512,alignment:{source:'native',mode:'live',level:'word'}}));
 socket.send(JSON.stringify({type:'text',text:item.text}));socket.send(JSON.stringify({type:'end'}));
 const deadline=performance.now()+120000;
 while((!ended||r.pending)&&!failed&&performance.now()<deadline)await sleep(20);
 if(!ended||r.pending)failed=failed||{message:'Playback deadline exceeded'};
 socket.close();
 }
 await sleep(220);
 const outputTimestamp=ctx.getOutputTimestamp?.(),outputLatency=ctx.outputLatency,baseLatency=ctx.baseLatency;
 const duration=ctx.currentTime,sampleRate=ctx.sampleRate,state=ctx.state;
 await ctx.close();
 const firstFrame=blocks.length?Math.min(...blocks.map(b=>b.start)):0;
 const captureOverlaps=[];
 const endFrame=blocks.reduce((end,b)=>Math.max(end,b.start+b.samples.length),firstFrame);
 const rendered=new Float32Array(endFrame-firstFrame);let offset=0,captureGapFrames=0,lastEnd=firstFrame;
 for(const b of blocks){
  if(b.start<lastEnd)captureOverlaps.push({start:b.start,previous_end:lastEnd,frames:lastEnd-b.start});
  captureGapFrames+=Math.max(0,b.start-lastEnd);rendered.set(b.samples,b.start-firstFrame);lastEnd=Math.max(lastEnd,b.start+b.samples.length);
 }
 if(captureOverlaps.length)failed=failed||{message:'Overlapping AudioWorklet capture blocks; raw evidence preserved'};
 let first=-1,last=-1;
 for(let i=0;i<rendered.length;i++)if(Math.abs(rendered[i])>=256/32768){if(first<0)first=i;last=i;}
 const received=new Uint8Array(recorded.reduce((s,b)=>s+b.byteLength,0));offset=0;
 for(const b of recorded){received.set(new Uint8Array(b),offset);offset+=b.byteLength;}
 let binary='';for(let i=0;i<received.length;i+=32768)binary+=String.fromCharCode(...received.subarray(i,i+32768));
 const result={id:item.id,text:item.text,voice:item.voice,language:item.language,seed:item.seed??20261009+index,
  kind:item.controlled?'controlled_pcm_real_webaudio':'native_websocket_real_webaudio',
  events,error:failed,sample_rate:sampleRate,source_sha256:CONFIG.source_sha256,context_priming_ms:100,
  controlled:measureControlled(r,rendered,blocks[0]?.start,sampleRate,received),
  rendered_start_frame:blocks[0]?.start,rendered_pcm_b64:encodePCM(rendered),received_pcm_b64:btoa(binary),
  capture_gap_frames:captureGapFrames,capture_overlaps:captureOverlaps,
  capture_blocks:blocks.map(b=>({start:b.start,frames:b.samples.length})),
  raw_capture_pcm_b64:encodePCM(Float32Array.from(blocks.flatMap(b=>Array.from(b.samples)))),
  rendered_first_signal_frame:first,rendered_last_signal_frame:last,
  audio_context_state:state,context_duration:duration,outputTimestamp,outputLatency,baseLatency,
  request_started_performance_ms:started,first_rendered_clock:firstRenderedClock,
  first_signal_output_clock_ms:firstRenderedClock?.output_timestamp?.performanceTime>0?
   firstRenderedClock.output_timestamp.performanceTime+
    (firstRenderedClock.sample_frame/sampleRate-firstRenderedClock.output_timestamp.contextTime)*1000-started:null,
  latency_basis:'Rendered PCM mapped to the browser output clock; no acoustic loopback measurement.',
  first_queue_metric:$('pcmMetric').textContent,first_audible_estimate:$('audibleMetric').textContent,
  underruns:r.underruns,gap_ms:r.gapMs,received_samples:r.samples,
  segments:r.playbackSegments.map(s=>({sampleStart:s.sampleStart,sampleCount:s.sampleCount,
   when:s.when,guardStopAt:s.guardStopAt})),wall_ms:performance.now()-started};
 const saved=await fetch('/result',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(result)});
 if(!saved.ok)throw Error('Result could not be saved');
 return {id:item.id,error:failed,received_seconds:r.samples/24000,verified:events.some(e=>e.type==='alignment_complete'),
  underruns:r.underruns,rendered_signal:first>=0,wall_ms:result.wall_ms,
  controlled:result.controlled};
}
document.getElementById('start').onclick=async()=>{
 const button=document.getElementById('start'),status=document.getElementById('status');button.disabled=true;
 window.benchmarkResults=[];window.benchmarkState='running';
 try{
  for(let i=0;i<CONFIG.cases.length;i++){
   status.textContent='Playing '+(i+1)+' / '+CONFIG.cases.length+': '+CONFIG.cases[i].id;
   const result=await checkCase(CONFIG.cases[i],i);window.benchmarkResults.push(result);
   document.getElementById('results').textContent=JSON.stringify(window.benchmarkResults,null,2);
  }
  const failures=window.benchmarkResults.filter(r=>r.error||r.controlled?.passed===false).length;
  window.benchmarkState='complete';status.textContent='Complete: '+window.benchmarkResults.length+' cases, '+failures+' errors';
 }catch(error){window.benchmarkState='failed';window.benchmarkError=String(error);status.textContent='Failed: '+error;}
};
window.benchmarkState='ready';
"""


def controlled_cases(path):
    """Use complete captured PCM as the intended region; appended speech is surplus.

    These cases test the browser scheduler, independently of recognition quality.
    They do not claim to be native engine or acoustic-device measurements.
    """
    raw = path.read_bytes()
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 24000):
            raise ValueError("Controlled audio must be mono PCM16 at 24000 Hz")
        frames = wav.getnframes()
        pcm = wav.readframes(frames)
    if not frames or len(pcm) != frames * 2:
        raise ValueError("Controlled audio must contain complete PCM frames")
    modes = ("original", "repeat-control", "queued-cutoff", "late-cutoff",
             "stale-cutoff", "following-fragment", "following-late-metadata")
    cases = [{"id": "controlled-" + mode, "controlled": mode,
              "fixture_frames": frames,
              "fixture_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
              "fixture_wav_sha256": hashlib.sha256(raw).hexdigest(),
              "fixture_source": str(path.resolve())} for mode in modes]
    return cases, pcm

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url",required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--port",type=int,default=18769)
    parser.add_argument("--api-key-env",default="REALTIME_TTS_API_KEY")
    parser.add_argument("--ids",nargs="*")
    parser.add_argument("--controlled-audio",type=Path,help="Uncensored mono PCM16 WAV at 24 kHz")
    parser.add_argument("--controlled-only",action="store_true")
    args=parser.parse_args()
    key=os.environ.get(args.api_key_env,"")
    cases=json.loads(args.manifest.read_text(encoding="utf-8"))["cases"]
    if args.ids:cases=[c for c in cases if c["id"] in args.ids]
    if args.controlled_only:cases=[]
    if cases and not key:parser.error("Native cases require the API key environment variable")
    fixture_pcm=None
    if args.controlled_audio:
        controls,fixture_pcm=controlled_cases(args.controlled_audio)
        cases.extend(controls)
    if not cases:parser.error("No playback cases selected")
    args.output.mkdir(parents=True,exist_ok=False)
    studio=(ROOT/"RealtimeTTS/studio/studio.js").read_text(encoding="utf-8")
    helpers=studio[studio.index("function playbackGuardState("):studio.index("function receivePCM(")]
    receiver=next(line for line in studio.splitlines() if line.startswith("function receivePCM("))
    source_sha256={"studio":hashlib.sha256(studio.encode()).hexdigest(),
                   "harness":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config=json.dumps({"base":args.base_url.rstrip("/"),"key":key,"cases":cases,"source_sha256":source_sha256},ensure_ascii=True).replace("<","\\u003c")
    script="const CONFIG="+config+";\n"+CONTROLLED+"\n"+HARNESS+"\n"+helpers+"\n"+receiver
    html=("<!doctype html><meta charset=utf-8><title>Native TTS playback validation</title>"
          "<h1>Native TTS playback validation</h1><p>Uses the actual Studio scheduler, WebSocket transport, and browser audio engine.</p>"
          "<button id=start>Run playback checks</button><p id=status>Ready</p><pre id=results></pre><script src=/harness.js></script>")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path=="/fixture.pcm" and fixture_pcm is not None:
                self.send_response(200);self.send_header("Content-Type","application/octet-stream")
                self.send_header("Content-Length",str(len(fixture_pcm)));self.end_headers();self.wfile.write(fixture_pcm)
                return
            content,kind=({"":(html,"text/html"),"/":(html,"text/html"),
                           "/harness.js":(script,"text/javascript"),
                           "/worklet.js":(WORKLET,"text/javascript")}.get(self.path,(None,None)))
            if content is None:self.send_error(404);return
            data=content.encode();self.send_response(200);self.send_header("Content-Type",kind)
            self.send_header("Cache-Control","no-store");self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
        def do_POST(self):
            if self.path!="/result":self.send_error(404);return
            length=int(self.headers.get("Content-Length","0"))
            if not 0<length<32*1024*1024:self.send_error(413);return
            value=json.loads(self.rfile.read(length));name=value["id"]
            if not re.fullmatch(r"[a-zA-Z0-9_-]+",name):self.send_error(400);return
            for field in ("rendered_pcm_b64","received_pcm_b64","raw_capture_pcm_b64"):
                raw=base64.b64decode(value.pop(field),validate=True)
                with (args.output/(name+"-"+field.removesuffix("_b64")+".pcm")).open("xb") as f:f.write(raw)
            with (args.output/(name+".json")).open("x",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False,indent=2)
            self.send_response(201);self.end_headers()
    print(json.dumps({"url":f"http://127.0.0.1:{args.port}","cases":len(cases),"output":str(args.output)}),flush=True)
    ThreadingHTTPServer(("127.0.0.1",args.port),Handler).serve_forever()

if __name__=="__main__":main()
