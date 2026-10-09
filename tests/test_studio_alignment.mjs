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

function studio() {
  const controls = new Map();
  const initial = {
    voice: 'generic_voice', language: 'en', transport: 'ws',
    alignmentSource: '', alignmentLevel: 'word', alignmentMode: 'live',
    alignmentBufferSeconds: '', temperature: '0.9', top_p: '1', top_k: '50',
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
function json(s, expression) {
  return JSON.parse(JSON.stringify(s.run(expression)));
}

test('generic capability fallback omits unsupported controls and serializes alignment', () => {
  const s = studio();
  s.run("caps=normalizeCapabilities({engine:{name:'generic',model_id:'generic'},model:{id:'generic'},features:{model_type:'base'}});defaults()");
  assert.equal(s.run('isQwen()'), false);
  assert.equal(s.run("samplingSupported('temperature')"), false);
  assert.equal(s.run('cloningSupported()'), false);
  s.run('applyMode()');
  assert.equal(s.control('cloneControls').hidden, true);
  assert.equal(s.control('voicePanel').hidden, true);
  assert.equal(s.control('temperature').hidden, true);
  s.control('alignmentSource').value = 'auto';
  assert.deepEqual(json(s, 'requestOptions().alignment'), {source: 'auto', level: 'word', mode: 'live'});
  assert.equal(json(s, "Object.hasOwn(requestOptions(),'temperature')"), false);
  s.control('alignmentLevel').value = 'character';
  s.control('alignmentMode').value = 'synced';
  s.control('alignmentBufferSeconds').value = '0.5';
  assert.deepEqual(json(s, 'requestOptions().alignment'), {
    source: 'auto', level: 'character', mode: 'synced', max_buffer_seconds: 0.5,
  });
  s.control('alignmentMode').value = 'export';
  assert.deepEqual(json(s, 'requestOptions().alignment'), {
    source: 'auto', level: 'character', mode: 'export',
  });
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
  assert.ok(s.control('alignmentText').children.some(x =>
    x.className === 'alignment-token' && x.textContent === '\u{1F600}'));
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
