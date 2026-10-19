import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';

const source = fs.readFileSync(
  path.join(path.dirname(fileURLToPath(import.meta.url)), '../RealtimeTTS/studio/studio.js'),
  'utf8',
);
const html = fs.readFileSync(
  path.join(path.dirname(fileURLToPath(import.meta.url)), '../RealtimeTTS/studio/index.html'),
  'utf8',
);

function studio() {
  const controls = new Map();
  const initial = {
    voice: 'generic_voice', language: 'en', transport: 'ws',
    temperature: '0.9', top_p: '1', top_k: '50',
    seed: '-1', max_new_tokens: '2048', repetition_penalty: '1.05',
    subtalker_temperature: '', subtalker_top_p: '', subtalker_top_k: '',
    subtalker_do_sample: '',
  };
  const canvasContext = {clearRect() {}, fillRect() {}, beginPath() {}, moveTo() {},
    lineTo() {}, stroke() {}, scale() {}, setLineDash() {}, save() {}, restore() {}};
  function node(id = '') {
    let value = initial[id] ?? '';
    const item = {
      id, children: [], checked: true, disabled: false, hidden: false,
      textContent: '', className: '', dataset: {}, options: [],
      clientWidth: 600, width: 600, height: 108,
      classList: {toggle() {}, add() {}, remove() {}},
      replaceChildren(...items) {
        this.children = items;
        this.textContent = items.map(x => x?.textContent ?? '').join('');
      },
      append(...items) {
        this.children.push(...items);
        this.textContent = this.children.map(x => x?.textContent ?? '').join('');
      },
      appendChild(x) { this.append(x); return x; },
      addEventListener() {},
      closest() { return this; },
      getBoundingClientRect() { return {left: 0, width: this.clientWidth}; },
      getContext() { return canvasContext; },
      checkValidity() { return true; },
      dispatchEvent() {},
    };
    Object.defineProperty(item, 'value', {
      get() { return value; },
      set(next) { value = next == null ? '' : String(next); },
    });
    return item;
  }
  const control = id => {
    if (!controls.has(id)) controls.set(id, node(id));
    return controls.get(id);
  };
  class FakeAudioContext {
    constructor() { this.state = 'suspended'; this.currentTime = 0; this.destination = {}; }
    resume() { this.state = 'running'; return Promise.resolve(this); }
    close() { this.state = 'closed'; return Promise.resolve(); }
  }
  class FakeWebSocket {
    static OPEN = 1;
    constructor() { this.readyState = 0; }
    send() {}
    close() {}
  }
  const context = vm.createContext({
    AudioContext: FakeAudioContext, WebSocket: FakeWebSocket, URL, Blob,
    AbortController, Option: class { constructor(text, value) {
      this.textContent = text; this.value = value;
    }},
    document: {
      getElementById: control,
      querySelectorAll: () => [],
      querySelector: () => ({value: 'speaker_only', checked: true, dispatchEvent() {}}),
      createElement: tag => Object.assign(node('created-' + tag), {type: tag}),
      createTextNode: text => ({textContent: String(text)}),
    },
    location: {origin: 'http://localhost'}, devicePixelRatio: 1,
    performance: {now: () => 0}, window: {addEventListener() {}},
    Event: class {}, requestAnimationFrame: undefined,
    cancelAnimationFrame: undefined, fetch: async () => ({ok: false}),
    confirm: () => true,
  });
  vm.runInContext(source, context);
  return {context, control, run: expression => vm.runInContext(expression, context)};
}

function setRun(s, run) {
  s.context.tempRun = run;
  s.run('run=tempRun');
}
function send(s, run, value) {
  s.context.tempRun = run;
  s.context.tempEvent = value;
  return s.run('handleAlignmentEvent(tempRun,tempEvent)');
}

test('verified delayed guard trims only its fragment with alignment display disabled', () => {
  const s = studio(), stopped = [];
  const r = {samples: 0, ctx: {currentTime: .2}, playbackSegments: []};
  setRun(s, r);
  s.context.guardRun = r;
  const dispatch = event => {
    s.context.guardEvent = event;
    return s.run('handlePlaybackGuardEvent(guardRun,guardEvent)');
  };
  const identity = {session_id: 's', request_id: 'first', fragment_index: 0};
  assert.equal(dispatch({...identity,type:'fragment_start',sample_rate_hz:24000}),true);
  const first = {sampleStart:0,sampleCount:48000,sampleRate:24000,when:1,
    source:{stop:when=>stopped.push(['first',when])}};
  r.playbackSegments.push(first);
  s.run('playbackGuardState(guardRun).wireBytes=96000');
  assert.equal(dispatch({...identity,type:'fragment_end'}),true);
  const secondId = {...identity,request_id:'second',fragment_index:1};
  assert.equal(dispatch({...secondId,type:'fragment_start',sample_rate_hz:24000}),true);
  r.playbackSegments.push({sampleStart:48000,sampleCount:24000,sampleRate:24000,when:3,
    source:{stop:when=>stopped.push(['second',when])}});
  const complete = {...identity,type:'alignment_complete',stop_requested:true,
    completion_verified:true,end_time:.5,trim_end_time:.75};
  for (const patch of [{stop_requested:false},{completion_verified:false},
                       {request_id:'second'},{session_id:'stale'},
                       {trim_end_time:.6},{trim_end_time:NaN}]) {
    assert.equal(dispatch({...complete,...patch}),false);
  }
  assert.deepEqual(stopped,[]);
  assert.equal(dispatch(complete),true);
  assert.deepEqual(stopped,[['first',1.75]]);
  // Repeated metadata cannot reopen audio or affect a successor.
  assert.equal(dispatch({...complete,trim_end_time:1.5}),true);
  assert.deepEqual(stopped,[['first',1.75]]);
  const future = {sampleStart:30000,sampleCount:4800,sampleRate:24000,when:2.25,
    source:{stop:when=>stopped.push(['future',when])}};
  s.context.futureGuardSegment = future;
  s.run('applyPlaybackGuard(guardRun,futureGuardSegment)');
  assert.deepEqual(stopped.at(-1),['future',2.25]);
  setRun(s,{ctx:r.ctx});
  assert.equal(dispatch(complete),false);
});

test('guard tracks wire offsets before AudioContext is ready and fences following fragments', () => {
  const s = studio();
  const r = {samples:0,ctx:{currentTime:0},audioReady:false,preReady:[]};
  setRun(s,r);
  s.context.guardRun=r;
  const dispatch = event => {
    s.context.guardEvent=event;
    return s.run('handlePlaybackGuardEvent(guardRun,guardEvent)');
  };
  const first={session_id:'s',request_id:'first',fragment_index:0};
  dispatch({...first,type:'fragment_start',sample_rate_hz:24000});
  s.run('receivePCM(guardRun,new ArrayBuffer(4800),0)');
  assert.equal(r.samples,0);
  assert.equal(r.preReady.length,1);
  dispatch({...first,type:'fragment_end'});
  dispatch({session_id:'s',request_id:'second',fragment_index:1,
    type:'fragment_start',sample_rate_hz:24000});
  assert.equal(r.playbackGuard.fragments.get(0).endSample,2400);
  assert.equal(r.playbackGuard.fragments.get(1).startSample,2400);
});

test('a guard received after the cutoff cannot pretend to retract played samples', () => {
  const s = studio(), stops=[];
  const r={ctx:{currentTime:2},playbackSegments:[]};
  setRun(s,r);
  s.context.guardRun=r;
  s.run("handlePlaybackGuardEvent(guardRun,{type:'fragment_start',session_id:'s',request_id:'r',fragment_index:0,sample_rate_hz:24000})");
  r.playbackSegments.push({sampleStart:0,sampleCount:72000,sampleRate:24000,when:0,
    source:{stop:when=>stops.push(when)}});
  s.run("handlePlaybackGuardEvent(guardRun,{type:'alignment_complete',session_id:'s',request_id:'r',fragment_index:0,stop_requested:true,completion_verified:true,end_time:.75,trim_end_time:1})");
  assert.deepEqual(stops,[2]);
});


test('PCM scheduling applies a verified cutoff with a short fade and no startup reserve', () => {
  const s = studio(), starts=[],stops=[],ramps=[];
  const ctx={currentTime:0,destination:{},
    createBuffer:()=>({copyToChannel(){}}),
    createGain:()=>({connect(){},disconnect(){},gain:{
      cancelScheduledValues(){},setValueAtTime(){},
      linearRampToValueAtTime(value,when){ramps.push([value,when]);}}}),
    createBufferSource:()=>({connect(){},disconnect(){},
      start:when=>starts.push(when),stop:when=>stops.push(when)})};
  const r={ctx,samples:0,next:0,odd:null,pending:0,sources:new Set(),audioReady:true,
    preReady:[],started:0,audible:false,underruns:0,gapMs:0,alignment:{enabled:false}};
  setRun(s,r);
  s.context.guardRun=r;
  s.run("handlePlaybackGuardEvent(guardRun,{type:'fragment_start',session_id:'s',request_id:'r',fragment_index:0,sample_rate_hz:24000})");
  s.run('receivePCM(guardRun,new ArrayBuffer(7680),0);receivePCM(guardRun,new ArrayBuffer(7680),0)');
  assert.deepEqual(starts,[0,.16]);
  ctx.currentTime=.1;
  s.run("handlePlaybackGuardEvent(guardRun,{type:'alignment_complete',session_id:'s',request_id:'r',fragment_index:0,stop_requested:true,completion_verified:true,end_time:.05,trim_end_time:.3})");
  assert.equal(stops.length,1);
  assert.ok(Math.abs(stops[0]-.3)<1e-9);
  assert.equal(ramps.length,1);
  assert.equal(ramps[0][0],0);
  assert.ok(Math.abs(ramps[0][1]-.3)<1e-9);
  assert.equal(r.samples,7680);
  assert.equal(r.playbackGuard.wireBytes,15360);
});


for (const timing of ['before-next','queued-next','already-playing-next']) {
  test('verified trim reclaims queued time: '+timing, () => {
    const s=studio(),sources=[];
    const ctx={currentTime:0,destination:{},
      createBuffer:(_,frames)=>({frames,copyToChannel(){}}),
      createGain:()=>({connect(){},disconnect(){},gain:{
        cancelScheduledValues(){},setValueAtTime(){},linearRampToValueAtTime(){}}}),
      createBufferSource:()=>{const node={connect(){},disconnect(){},
        start(when){this.started=when;},stop(when){this.stopped=when??ctx.currentTime;}};
        sources.push(node);return node;}};
    const r={ctx,samples:0,next:0,odd:null,pending:0,sources:new Set(),audioReady:true,
      preReady:[],started:0,audible:false,underruns:0,gapMs:0,alignment:{enabled:false}};
    setRun(s,r);s.context.guardRun=r;
    const dispatch=event=>{s.context.guardEvent=event;return s.run('handlePlaybackGuardEvent(guardRun,guardEvent)');};
    const first={session_id:'s',request_id:'first',fragment_index:0};
    const second={session_id:'s',request_id:'second',fragment_index:1};
    dispatch({...first,type:'fragment_start',sample_rate_hz:24000});
    s.run('receivePCM(guardRun,new ArrayBuffer(24000),0);receivePCM(guardRun,new ArrayBuffer(72000),0)');
    dispatch({...first,type:'fragment_end'});
    const cutoff={...first,type:'alignment_complete',stop_requested:true,
      completion_verified:true,end_time:.5,trim_end_time:.75};
    ctx.currentTime=.1;
    if(timing==='before-next')dispatch(cutoff);
    dispatch({...second,type:'fragment_start',sample_rate_hz:24000});
    s.run('receivePCM(guardRun,new ArrayBuffer(48000),0)');
    dispatch({...second,type:'fragment_end'});
    const previousSource=r.playbackSegments.at(-1).source,buffer=previousSource.buffer;
    if(timing==='already-playing-next')ctx.currentTime=2.2;
    if(timing!=='before-next')dispatch(cutoff);
    const next=r.playbackSegments.at(-1),expected=timing==='already-playing-next'?2:.75;
    assert.equal(next.when,expected);
    assert.equal(next.source.buffer,buffer);
    assert.equal(next.sampleCount,24000);
    assert.equal(r.next,expected+1);
    assert.equal(r.pending,3);assert.equal(r.sources.size,3);
    if(timing==='queued-next'){
      assert.notEqual(next.source,previousSource);assert.equal(previousSource.onended,null);
      assert.equal(previousSource.stopped,.1);
    }
    dispatch(cutoff);assert.equal(r.next,expected+1);
    ctx.currentTime=expected+.1;
    assert.equal(s.run('alignmentClockSample({playbackSegments:guardRun.playbackSegments},guardRun.ctx)'),50400);
  });
}

function json(s, expression) {
  return JSON.parse(JSON.stringify(s.run(expression)));
}

test('studio keeps live alignment inside the editable speech window', () => {
  assert.match(html, /id="textHighlight"/);
  assert.doesNotMatch(html, /id="alignmentPanel"/);
  assert.doesNotMatch(html, /id="alignmentLevel"/);
});

test('generic capability fallback omits unsupported controls and prefers character alignment', () => {
  const s = studio();
  s.run("caps=normalizeCapabilities({engine:{name:'generic',model_id:'generic'},model:{id:'generic'},features:{model_type:'base'}});defaults()");
  assert.equal(s.run('isQwen()'), false);
  assert.equal(s.run("samplingSupported('temperature')"), false);
  assert.equal(s.run('cloningSupported()'), false);
  s.run('applyMode()');
  assert.equal(s.control('cloneControls').hidden, true);
  assert.equal(s.control('voicePanel').hidden, true);
  assert.equal(s.control('temperature').hidden, true);
  assert.deepEqual(json(s, 'requestOptions().alignment'), {source: 'auto', level: 'character', mode: 'live'});
  assert.equal(json(s, "Object.hasOwn(requestOptions(),'temperature')"), false);
  s.run("caps.features.alignment_timestamps=['word'];");
  assert.deepEqual(json(s, 'requestOptions().alignment'), {source: 'auto', level: 'word', mode: 'live'});
  s.run("caps.features.alignment_timestamps=[];");
  assert.equal(json(s, "Object.hasOwn(requestOptions(),'alignment')"), false);
});

test('speed stays configurable while live character alignment is automatic', () => {
  const s = studio();
  s.run("caps=normalizeCapabilities({engine:{name:'generic',model_id:'generic'},model:{id:'generic'},features:{model_type:'base'},playback:{speed:{min:0.25,max:4,default:1}}});defaults()");
  s.control('speed').value = '1.5';
  assert.equal(json(s, 'requestOptions().speed'), 1.5);
  assert.deepEqual(json(s, 'requestOptions().alignment'), {source: 'auto', level: 'character', mode: 'live'});
  s.control('speed').value = '0.1';
  assert.throws(() => s.run('requestOptions()'), /Speed must be between/);
});

test('alignment revisions, code-point spans, old deltas, and stale runs are fenced', () => {
  const s = studio();
  const opts = {source: 'auto', level: 'character', mode: 'live'};
  const state = s.run('createAlignmentState(' + JSON.stringify(opts) + ')');
  const run = {alignment: state, samples: 0, done: false, ctx: null};
  setRun(s, run);
  assert.equal(send(s, run, {type: 'fragment_start', request_id: 'r1', fragment_index: 0,
    text: 'A\u{1F600}B', sample_rate_hz: 100}), true);
  assert.equal(send(s, run, {type: 'alignment', request_id: 'r1', fragment_index: 0,
    revision: 1, characters: [
      {character: 'A', text_start: 0, text_end: 1, start_time: 0, end_time: .2},
      {character: '\u{1F600}', text_start: 1, text_end: 2, start_time: .2, end_time: .5},
      {character: 'B', text_start: 2, text_end: 3, start_time: .5, end_time: .7},
    ]}), true);
  assert.equal(send(s, run, {type: 'alignment', request_id: 'r1', fragment_index: 0,
    revision: 0, characters: [{character: 'A', text_start: 0, text_end: 1,
      start_time: 0, end_time: 9}]}), false);
  const fragment = state.fragments.get(0);
  assert.equal(fragment.characters[1].character, '\u{1F600}');
  assert.ok(s.control('textHighlight').children.some(x =>
    x.className === 'studio-alignment-token' && x.textContent === '\u{1F600}'));
  assert.equal(send(s, run, {type: 'alignment', request_id: 'r1', fragment_index: 0,
    characters: [{character: 'B', text_start: 2, text_end: 3, start_time: .6, end_time: .8}]}), true);
  assert.equal(fragment.characters[2].end_time, .8);
  assert.equal(send(s, run, {type: 'alignment_complete', request_id: 'r1',
    fragment_index: 0, revision: 2, status: 'ready', final: true}), true);
  assert.equal(state.status, 'ready');
  assert.equal(state.final, true);
  const timeoutState = s.run('createAlignmentState(' + JSON.stringify(opts) + ')');
  const timeoutRun = {alignment: timeoutState, samples: 0, done: false, ctx: null};
  setRun(s, timeoutRun);
  assert.equal(send(s, timeoutRun, {type: 'alignment_done', request_id: 'r-timeout',
    fragment_index: 0, complete: false, finish_reason: 'alignment_timeout'}), true);
  assert.equal(timeoutState.status, 'behind');
  assert.equal(timeoutState.final, false);
  const next = s.run('createAlignmentState(' + JSON.stringify(opts) + ')');
  setRun(s, {alignment: next, samples: 0, done: false, ctx: null});
  assert.equal(send(s, run, {type: 'alignment', request_id: 'r1',
    fragment_index: 0, revision: 3}), false);
});


test('overlay keeps measured word and character highlights on the same text', () => {
  const s = studio();
  const opts = {source: 'auto', level: 'character', mode: 'live'};
  const state = s.run('createAlignmentState(' + JSON.stringify(opts) + ')');
  const run = {alignment: state, samples: 0, done: false, ctx: {currentTime: .25}};
  setRun(s, run);
  send(s, run, {type: 'fragment_start', request_id: 'r1', fragment_index: 0,
    text: 'hello', sample_rate_hz: 100});
  send(s, run, {type: 'alignment', request_id: 'r1', fragment_index: 0, revision: 1,
    words: [{word: 'hello', text_start: 0, text_end: 5, start_time: .1, end_time: .4}],
    characters: [{character: 'e', text_start: 1, text_end: 2, start_time: .2, end_time: .3}]});
  state.playbackSegments = [{sampleStart: 0, sampleCount: 100, sampleRate: 100, when: 0}];
  s.run('renderAlignmentState(run.alignment);updateAlignmentHighlight(run)');
  const active = JSON.parse(s.run("JSON.stringify({word:run.alignment.overlayWords[0].tokens[0].dataset.alignmentActive,character:run.alignment.overlayCharacters[0].tokens[0].dataset.alignmentActive})"));
  assert.deepEqual(active, {word: 'true', character: 'true'});
  run.ctx.currentTime = .5;
  s.run('updateAlignmentHighlight(run)');
  assert.equal(s.run("run.alignment.overlayWords[0].tokens[0].dataset.alignmentComplete"), 'true');
});
test('edited speech text does not fabricate highlights from stale fragments', () => {
  const s = studio();
  const state = s.run('createAlignmentState(' + JSON.stringify({source: 'auto', level: 'word', mode: 'live'}) + ')');
  const run = {alignment: state, samples: 0, done: false, ctx: null};
  setRun(s, run);
  send(s, run, {type: 'fragment_start', request_id: 'r1', fragment_index: 0,
    text: 'original', sample_rate_hz: 100});
  send(s, run, {type: 'alignment', request_id: 'r1', fragment_index: 0, revision: 1,
    words: [{word: 'original', text_start: 0, text_end: 8, start_time: 0, end_time: .5}]});
  s.control('text').value = 'edited';
  s.run('renderAlignmentText(run.alignment)');
  assert.equal(state.overlayWords.length, 0);
  assert.equal(state.overlayCharacters.length, 0);
});
test('alignment accepts per-fragment request ids and fences stale session data', () => {
  const s = studio();
  const opts = {source: 'auto', level: 'word', mode: 'live'};
  const state = s.run('createAlignmentState(' + JSON.stringify(opts) + ')');
  const run = {alignment: state, samples: 0, done: false, ctx: null};
  setRun(s, run);
  assert.equal(send(s, run, {type: 'ready', session_id: 'session-1', request_id: 'session-1'}), false);
  assert.equal(state.sessionId, 'session-1');
  assert.equal(send(s, run, {type: 'fragment_start', session_id: 'session-1',
    request_id: 'fragment-0', fragment_index: 0, text: 'first', sample_rate_hz: 100}), true);
  assert.equal(send(s, run, {type: 'fragment_start', session_id: 'session-1',
    request_id: 'fragment-1', fragment_index: 1, text: 'second', sample_rate_hz: 100}), true);
  assert.equal(send(s, run, {type: 'alignment', session_id: 'session-1',
    request_id: 'fragment-1', fragment_index: 1, revision: 1,
    words: [{word: 'second', text_start: 0, text_end: 6, start_time: .2, end_time: .6}]}), true);
  assert.equal(send(s, run, {type: 'alignment', session_id: 'session-1',
    request_id: 'fragment-0', fragment_index: 0, revision: 1,
    words: [{word: 'first', text_start: 0, text_end: 5, start_time: 0, end_time: .2}]}), true);
  assert.equal(state.requestIds.get(0), 'fragment-0');
  assert.equal(state.requestIds.get(1), 'fragment-1');
  assert.equal(state.fragments.get(0).words[0].word, 'first');
  assert.equal(state.fragments.get(1).words[0].word, 'second');
  assert.equal(send(s, run, {type: 'alignment', session_id: 'session-1',
    request_id: 'stale-fragment-0', fragment_index: 0, revision: 2,
    words: [{word: 'wrong', text_start: 0, text_end: 5, start_time: 0, end_time: 9}]}), false);
  assert.equal(state.fragments.get(0).words[0].word, 'first');
  assert.equal(send(s, run, {type: 'alignment', session_id: 'session-2',
    request_id: 'fragment-1', fragment_index: 1, revision: 2,
    words: [{word: 'wrong-session', text_start: 0, text_end: 6, start_time: 0, end_time: 9}]}), false);
  assert.equal(state.fragments.get(1).words[0].word, 'second');
});

test('synced alignment adds its configured playback lead at schedule time', () => {
  const s = studio();
  assert.equal(s.run("alignmentPlaybackLeadSeconds({enabled:true,options:{mode:'synced'}})"), .25);
  assert.equal(s.run("alignmentPlaybackLeadSeconds({enabled:true,options:{mode:'synced',max_buffer_seconds:0}})"), 0);
  assert.equal(s.run("alignmentPlaybackLeadSeconds({enabled:true,options:{mode:'synced',max_buffer_seconds:.5}})"), .5);
  assert.equal(s.run("alignmentPlaybackLeadSeconds({enabled:true,options:{mode:'live',max_buffer_seconds:.5}})"), 0);
  const scheduled = [];
  const context = {
    currentTime: 10, destination: {},
    createBuffer: () => ({copyToChannel() {}}),
    createBufferSource: () => ({
      connect() {}, disconnect() {},
      start(when) { scheduled.push(when); },
    }),
  };
  const state = s.run("createAlignmentState({source:'auto',level:'word',mode:'synced'})");
  const run = {alignment: state, samples: 0, done: false, ctx: context,
    audioReady: true, preReady: [], next: 0, odd: null, started: 0,
    lastPCM: 0, audible: false, underruns: 0, gapMs: 0,
    pending: 0, sources: new Set(), playbackSegments: []};
  setRun(s, run);
  s.control('bufferMs').value = '100';
  s.run('recorded=[]');
  s.context.tempRun = run;
  s.run("receivePCM(tempRun,new Uint8Array([0,64]).buffer,10)");
  assert.equal(scheduled.length, 1);
  assert.ok(Math.abs(scheduled[0] - 10.35) < 1e-9);
});

test('pause stays local when the engine has no native pause support', async () => {
  const s = studio();
  const messages = [];
  const context = {
    state: 'running', currentTime: 0,
    async suspend() { this.state = 'suspended'; },
    async resume() { this.state = 'running'; },
  };
  const run = {ctx: context, done: false,
    ws: {readyState: 1, send(value) { messages.push(JSON.parse(value)); }}};
  setRun(s, run);
  s.run('caps=normalizeCapabilities({stream_controls:{native_pause_resume:false}})');
  await s.run('pause()');
  assert.equal(context.state, 'suspended');
  assert.equal(s.control('playState').textContent, 'Paused');
  await s.run('pause()');
  assert.equal(context.state, 'running');
  assert.deepEqual(messages, []);

  s.run('caps.stream_controls.native_pause_resume=true');
  await s.run('pause()');
  await s.run('pause()');
  assert.deepEqual(messages, [{type: 'pause'}, {type: 'resume'}]);
});

test('subtitle timestamp rounding carries across seconds, minutes, and hours', () => {
  const s = studio();
  const cases = [
    [0, '00:00:00.000'],
    [-1, '00:00:00.000'],
    [NaN, '00:00:00.000'],
    ['1.234', '00:00:01.234'],
    [0.0005, '00:00:00.001'],
    [0.9994, '00:00:00.999'],
    [0.9996, '00:00:01.000'],
    [59.9996, '00:01:00.000'],
    [3599.9996, '01:00:00.000'],
    [86399.9996, '24:00:00.000'],
  ];
  for (const separator of [',', '.']) {
    for (const [seconds, expected] of cases) {
      s.context.timestampInput = seconds;
      s.context.timestampSeparator = separator;
      assert.equal(
        s.run('formatAlignmentTimestamp(timestampInput,timestampSeparator)'),
        expected.replace('.', separator),
        String(seconds),
      );
    }
  }
});

test('SRT and VTT exports carry rounded cue timestamps into the next second', () => {
  const s = studio();
  const state = s.run('createAlignmentState({source:"auto",level:"word",mode:"export"})');
  const run = {alignment: state, samples: 0, done: false, ctx: null};
  setRun(s, run);
  send(s, run, {type: 'fragment_start', request_id: 'r', fragment_index: 0,
    text: 'Hello', sample_rate_hz: 24000});
  send(s, run, {type: 'alignment', request_id: 'r', fragment_index: 0,
    revision: 1, words: [{word: 'Hello', text_start: 0, text_end: 5,
      start_time: .9996, end_time: 1.9996}]});
  s.context.exportState = state;
  assert.match(s.run("alignmentExportText('srt',exportState)"),
    /00:00:01,000 --> 00:00:02,000/);
  assert.match(s.run("alignmentExportText('vtt',exportState)"),
    /00:00:01\.000 --> 00:00:02\.000/);
});

test('playback clock and exports retain inter-fragment silence', () => {
  const s = studio();
  const opts = {source: 'auto', level: 'word', mode: 'live'};
  const state = s.run('createAlignmentState(' + JSON.stringify(opts) + ')');
  const run = {alignment: state, samples: 0, done: false, ctx: null};
  setRun(s, run);
  send(s, run, {type: 'fragment_start', request_id: 'r2', fragment_index: 0,
    text: 'Hello', sample_rate_hz: 100});
  send(s, run, {type: 'alignment', request_id: 'r2', fragment_index: 0,
    revision: 1, words: [{word: 'Hello', text_start: 0, text_end: 5,
      start_time: 0, end_time: .5}]});
  send(s, run, {type: 'fragment_start', request_id: 'r2', fragment_index: 1,
    text: 'world', sample_rate_hz: 100});
  send(s, run, {type: 'alignment', request_id: 'r2', fragment_index: 1,
    revision: 1, words: [{word: 'world', text_start: 0, text_end: 5,
      start_time: .2, end_time: .7}]});
  const first = state.fragments.get(0), second = state.fragments.get(1);
  state.firstPlaybackWhen = 1;
  first.pcmStartSample = 0; first.playbackStartWhen = 1;
  second.pcmStartSample = 100; second.playbackStartWhen = 3;
  state.playbackSegments = [
    {sampleStart: 0, sampleCount: 100, when: 1, sampleRate: 100},
    {sampleStart: 100, sampleCount: 100, when: 3, sampleRate: 100},
  ];
  state.totalSamples = 200;
  setRun(s, run);
  s.run('renderAlignmentState(run.alignment)');
  assert.equal(s.run('alignmentClockSample(run.alignment,{currentTime:3.2})'), 120);
  s.context.exportState = state;
  const snapshot = JSON.parse(s.run("alignmentExportText('json',exportState)"));
  assert.equal(snapshot.time_base, 'recording');
  assert.equal(snapshot.fragments[1].words[0].global_start_time, 2.2);
  assert.match(s.run("alignmentExportText('srt',exportState)"),
    /00:00:02,200 --> 00:00:02,700/);
  assert.match(s.run("alignmentExportText('vtt',exportState)"), /^WEBVTT/);
});


test('partial final alignment never displays Ready', () => {
  const s = studio();
  assert.equal(json(s, 'alignmentStatusFor({status:"partial",final:true})')[0], 'partial');
});


test('partial and unavailable completion remain final with an honest status', () => {
  for (const [payload, status] of [
    [{finish_reason: 'partial'}, 'partial'],
    [{status: 'partial'}, 'partial'],
    [{finish_reason: 'partial_alignment'}, 'partial'],
    [{complete: true, completion_verified: false, finish_reason: 'text_aligned_unverified'}, 'partial'],
    [{finish_reason: 'unavailable'}, 'unavailable'],
    [{error: 'model failed'}, 'unavailable'],
    [{cancelled: true}, 'cancelled'],
  ]) {
    const s = studio();
    const state = s.run('createAlignmentState({source:"model",level:"word",mode:"live"})');
    const run = {alignment: state, samples: 0, done: false, ctx: null};
    setRun(s, run);
    send(s, run, {type: 'fragment_start', request_id: 'r1', fragment_index: 0, text: 'hello world', sample_rate_hz: 100});
    send(s, run, {type: 'alignment_done', request_id: 'r1', fragment_index: 0, complete: false, ...payload});
    assert.equal(state.status, status);
    assert.equal(state.final, true);
  }
});
