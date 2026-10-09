import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const script = fs.readFileSync(new URL('../RealtimeTTS/studio/studio.js', import.meta.url), 'utf8');
const html = fs.readFileSync(new URL('../RealtimeTTS/studio/index.html', import.meta.url), 'utf8');
const prefix = script.slice(0, script.indexOf("\ndocument.querySelectorAll('.nav').forEach"));
function setup(breeze=true) {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id,{value:'',disabled:false,hidden:false,checked:true,
      textContent:'',options:[],min:'',dispatchEvent(){},checkValidity(){return true;},
      closest(){return this.label ||= {hidden:false};},replaceChildren(...items){this.children=items;}});
    return elements.get(id);
  }
  for (const match of html.matchAll(/<input\b[^>]*id="([^"]+)"[^>]*>/g)) {
    element(match[1]).value=match[0].match(/value="([^"]*)"/)?.[1] || '';
    element(match[1]).checked=/\bchecked\b/.test(match[0]);
  }
  element('voice').value='sl_ref1';element('language').value='en';
  element('instructions').value='Amused and conversational';element('synthesisMode').value='reference';
  const clone={value:'speaker_only',checked:true,dispatchEvent(){}};
  const context=vm.createContext({location:{origin:'http://localhost:18085'},console,
    document:{title:'',getElementById:element,querySelector:s=>s.includes('input[name=clone]')?clone:element('title'),querySelectorAll:()=>[]},
    performance:{now:()=>1100},devicePixelRatio:1,Event:class{},Option:class{constructor(text,value){this.text=text;this.value=value;}},
    setTimeout,clearTimeout,Blob,DataView,Float32Array,Uint8Array,ArrayBuffer,URL});
  vm.runInContext(prefix,context);
  const caps={model:{id:breeze?'breeze-tts-2-q8':'qwen'},engine:{name:breeze?'breeze':'qwen',clone_mode:breeze?'reference':'speaker_only'},
    features:{model_type:'base',instruction_control:true,sentence_splitting:true,voice_design:breeze,voice_design_voice:'__breeze_voice_design__',supported_languages:['en','zh'],vocal_tags:['laugh','chuckle']},
    sampling_defaults:{top_p:0,repetition_penalty:0,seed:44,max_new_tokens:500,cfg_scale:4}};
  context.fixture=caps;vm.runInContext('caps=fixture;',context);
  return {context,element,options:()=>JSON.parse(vm.runInContext('JSON.stringify(requestOptions())',context))};
}
test('Breeze reference payload excludes Qwen-only fields and accepts native zero defaults',()=>{
  const s=setup();vm.runInContext('defaults();applyMode();',s.context);
  const p=s.options();assert.equal(p.voice,'sl_ref1');assert.equal(p.top_p,0);assert.equal(p.cfg_scale,4);
  for(const field of ['clone_mode','do_sample','subtalker_do_sample','subtalker_temperature']) assert.ok(!(field in p));
  assert.equal(s.element('cloneControls').hidden,true);assert.equal(s.element('refText').required,true);
});
test('Breeze design combines the description and direction and sends no reference selection',()=>{
  const s=setup();s.element('synthesisMode').value='design';
  assert.throws(()=>s.options(),/Stimme beschreiben/);
  s.element('voiceDescription').value='A deep, resonant narrator.';
  const p=s.options();assert.equal(p.voice,'__breeze_voice_design__');
  assert.equal(p.instructions,'A deep, resonant narrator. Amused and conversational');
});
test('Breeze unsplit checkbox preserves defaults and reaches every transport payload',()=>{
  const s=setup();vm.runInContext('defaults();applyMode();',s.context);
  assert.equal(s.element('sentenceSplittingControl').hidden,false);
  assert.equal(s.options().split_sentences,true);
  s.element('noSentenceSplitting').checked=true;
  for(const transport of ['ws','pcm','wav']){
    s.element('transport').value=transport;
    assert.equal(s.options().split_sentences,false);
  }
  vm.runInContext("tabName='stream';applyMode();",s.context);
  assert.match(s.element('sentenceSplittingHint').textContent,/Abschliessen/);
  vm.runInContext('defaults();',s.context);
  assert.equal(s.options().split_sentences,true);
});
test('Older Breeze servers hide the unsupported control and Qwen requests stay unchanged',()=>{
  const s=setup();delete s.context.fixture.features.sentence_splitting;
  vm.runInContext('applyMode();',s.context);
  assert.equal(s.element('sentenceSplittingControl').hidden,true);
  assert.ok(!('split_sentences' in s.options()));
  const qwen=setup(false);vm.runInContext('applyMode();',qwen.context);
  assert.equal(qwen.element('sentenceSplittingControl').hidden,true);
  assert.ok(!('split_sentences' in qwen.options()));
});
test('Qwen payload keeps its original clone and codec controls',()=>{
  const s=setup(false),p=s.options();
  assert.equal(p.clone_mode,'speaker_only');assert.equal(p.do_sample,true);assert.equal(p.subtalker_temperature,0.9);
  assert.ok(!('cfg_scale' in p));
});
test('PCM starts without an additional browser reserve and subsequent chunks stay contiguous',()=>{
  const s=setup(),scheduled=[];
  assert.equal(s.element('bufferMs').value,'0');
  const ctx={currentTime:1,outputLatency:0,baseLatency:0,destination:{},
    createBuffer:(channels,length,rate)=>({copyToChannel(){},length,rate}),
    createBufferSource:()=>({connect(){},disconnect(){},start(when){scheduled.push(when);}})};
  s.context.fixtureRun={ctx,started:1000,odd:null,samples:0,pending:0,next:0,underruns:0,gapMs:0,audioReady:true,preReady:[],sources:new Set(),audible:false};
  s.context.chunk=new Int16Array(11520).fill(1000).buffer;
  vm.runInContext('drawWave=()=>{};run=fixtureRun;receivePCM(run,chunk);receivePCM(run,chunk);',s.context);
  assert.deepEqual(scheduled,[1,1.48]);assert.equal(s.context.fixtureRun.underruns,0);
});
