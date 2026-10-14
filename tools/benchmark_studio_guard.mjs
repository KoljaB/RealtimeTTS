// Replay measured engine queue/completion times through the actual Studio scheduler.
// This is an AudioContext clock model, not a browser or hardware playback measurement.
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const source = fs.readFileSync(path.join(root, 'RealtimeTTS/studio/studio.js'), 'utf8');
const helpers = source.slice(source.indexOf('function playbackGuardState('),
                             source.indexOf('function receivePCM('));
const receiver = source.split('\n').find(line => line.startsWith('function receivePCM('));
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const report = {kind:'studio_clock_model',input:process.argv[2],cases:[],
  limits:'Actual scheduler functions with a deterministic AudioContext model; no browser or device latency.'};

for (const item of input.cases) {
  const runData = item.runs.extra_speech;
  const controls = new Map();
  const ctx = {
    currentTime:0,destination:{},
    createBuffer(channels,count,rate){return {duration:count/rate,copyToChannel(){}};},
    createGain(){return {gain:{cancelScheduledValues(){},setValueAtTime(){},
      linearRampToValueAtTime(){}},connect(){},disconnect(){}};},
    createBufferSource(){return {connect(){},disconnect(){},
      start(when){this.startAt=when;},stop(when){this.stopAt=when;}};},
  };
  const r = {ctx,samples:0,next:0,odd:null,pending:0,sources:new Set(),preReady:[],
    audioReady:true,started:0,audible:false,underruns:0,gapMs:0,playbackSegments:[],
    alignment:{enabled:false}};
  const context = vm.createContext({run:r,recorded:[],performance:{now:()=>ctx.currentTime*1000},
    $:id=>{if(!controls.has(id))controls.set(id,{value:'0',textContent:''});return controls.get(id);},
    alignmentPlaybackLeadSeconds:()=>0,drawWave(){},maybeFinalize(){},
    stop(){throw Error('unexpected buffer limit');}});
  vm.runInContext(helpers+'\n'+receiver,context);
  const identity = {session_id:'capture',request_id:item.id,fragment_index:0};
  const dispatch = event => {
    context.event=event;
    return vm.runInContext('handlePlaybackGuardEvent(run,event)',context);
  };
  dispatch({...identity,type:'fragment_start',sample_rate_hz:24000});
  const events = runData.profile.queued_chunks.map(c=>({at:c.queued_ms,pcm:c.samples}));
  events.push({at:runData.profile.total_ms,event:{...identity,type:'fragment_end'}});
  if(runData.completion)events.push({at:runData.completion.wall_ms,
    event:{...runData.completion,...identity}});
  events.sort((a,b)=>a.at-b.at);
  for(const entry of events){
    ctx.currentTime=entry.at/1000;
    if(entry.pcm!==undefined){
      context.byteCount=entry.pcm*2;
      vm.runInContext('receivePCM(run,new ArrayBuffer(byteCount))',context);
    }else dispatch(entry.event);
  }
  const extraSample = Math.round(runData.extra_speech_onset_seconds*24000);
  let surplus=0;
  for(const segment of r.playbackSegments){
    const start=segment.when+Math.max(0,extraSample-segment.sampleStart)/24000;
    const end=Math.min(segment.when+segment.sampleCount/24000,
                       segment.source.stopAt??Infinity);
    surplus+=Math.max(0,end-start);
  }
  const record={id:item.id,verified:Boolean(runData.completion),
    surplus_received_seconds:Math.max(0,runData.output_seconds-runData.extra_speech_onset_seconds),
    surplus_played_clock_model_seconds:surplus,
    first_pcm_scheduled_seconds:r.playbackSegments[0]?.when};
  report.cases.push(record);
  console.log(JSON.stringify(record));
}
fs.writeFileSync(process.argv[3],JSON.stringify(report,null,2)+'\n',{flag:'wx'});
