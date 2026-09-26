'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../RealtimeTTS/studio/studio.js'), 'utf8');

function studio(codecDefault = null, device = 'cpu', options = {}) {
  const controls = new Map();
  const initial = {
    voice: 'reference', language: 'de', transport: 'ws', text: 'hello', temperature: '0.9', top_p: '1',
    top_k: '50', seed: '-1', max_new_tokens: '2048', repetition_penalty: '1.05',
    subtalker_temperature: '', subtalker_top_p: '', subtalker_top_k: '',
    subtalker_do_sample: ''
  };
  const canvas = new Proxy({}, {get: () => () => {}});
  const runtime = {contexts: [], sockets: []};
  class PlaybackSource {
    constructor(ctx) {
      this.ctx = ctx; this.startWhen = null; this.stopCalls = 0; this.disconnectCalls = 0;
      this.onended = null; ctx.sources.push(this);
    }
    connect() {}
    disconnect() { this.disconnectCalls += 1; }
    start(when) { this.startWhen = when; }
    stop() { this.stopCalls += 1; }
  }
  class PlaybackContext {
    constructor() {
      this.state = 'suspended'; this.currentTime = 0; this.destination = {};
      this.outputLatency = 0; this.baseLatency = 0; this.sources = [];
      this.resumeCalls = 0; this.closeCalls = 0; runtime.contexts.push(this);
    }
    resume() {
      this.resumeCalls += 1;
      if (options.deferResume) return new Promise(resolve => { this.resolveResume = () => {
        this.state = 'running'; resolve(this);
      }; });
      this.state = 'running';
      return Promise.resolve(this);
    }
    close() { this.closeCalls += 1; this.state = 'closed'; return Promise.resolve(); }
    createBuffer() { return {copyToChannel() {}}; }
    createBufferSource() { return new PlaybackSource(this); }
  }
  class FakeWebSocket {
    static OPEN = 1;
    constructor() { this.readyState = 0; this.sent = []; this.closeCalls = 0; runtime.sockets.push(this); }
    send(value) { this.sent.push(value); }
    close() { this.readyState = 3; this.closeCalls += 1; }
    open() { this.readyState = 1; this.onopen?.(); }
    message(data) { this.onmessage?.({data}); }
  }
  function control(id) {
    if (!controls.has(id)) {
      let value = initial[id] || '';
      controls.set(id, {
        get value() { return value; },
        set value(next) { value = next == null ? '' : String(next); },
        checked: true, disabled: id === 'instructions', clientWidth: 600, options: [],
        classList: {toggle() {}, add() {}, remove() {}},
        checkValidity: () => true, getContext: () => canvas,
        dispatchEvent() {}
      });
    }
    return controls.get(id);
  }
  const clone = {value: 'speaker_only', checked: true, dispatchEvent() {}};
  const listeners = {};
  const context = vm.createContext({
    AudioContext: PlaybackContext, WebSocket: FakeWebSocket, URL, AbortController,
    document: {getElementById: control, querySelectorAll: () => [], querySelector: () => clone},
    location: {origin: 'http://localhost'}, devicePixelRatio: 1, performance: {now: () => 0},
    window: {addEventListener(name, handler) { listeners[name] = handler; }}, Event: class {},
    fetch: async () => ({ok: false})
  });
  vm.runInContext(source, context);
  const caps = {
    model: {id: 'qwen'}, engine: {clone_mode: 'speaker_only', device},
    features: {model_type: 'base'},
    sampling_defaults: {
      temperature: .9, top_p: 1, top_k: 50, seed: -1, max_new_tokens: 2048,
      repetition_penalty: 1.05, do_sample: true, subtalker_do_sample: codecDefault,
      subtalker_temperature: null, subtalker_top_p: null, subtalker_top_k: null
    }
  };
  vm.runInContext('caps=' + JSON.stringify(caps) + '; defaults();', context);
  return {
    control,
    receiveAt: (arrivals) => {
      context.arrivals = arrivals;
      return JSON.parse(vm.runInContext(`
        (() => {
          const starts = [];
          const ctx = {
            currentTime: 0, destination: {},
            createBuffer: () => ({copyToChannel() {}}),
            createBufferSource: () => ({
              connect() {}, disconnect() {}, start(when) {starts.push(when);}
            })
          };
          const r = {ctx, pending: 0, samples: 0, next: 0, odd: null,
            started: 0, audible: false, underruns: 0, gapMs: 0, audioReady: true, preReady: [], sources: new Set()};
          run = r;
          for (const arrival of arrivals) {
            ctx.currentTime = arrival;
            receivePCM(r, new Int16Array(1920).buffer);
          }
          return JSON.stringify({starts, underruns: r.underruns, gapMs: r.gapMs});
        })()
      `, context));
    },
    reset: () => vm.runInContext('defaults()', context),
    start: () => vm.runInContext('start()', context),
    stop: () => vm.runInContext('stop()', context),
    context: () => runtime.contexts.at(-1),
    socket: () => runtime.sockets.at(-1),
    events: () => vm.runInContext('events', context),
    connectGesture: () => control('connectionForm').onsubmit({preventDefault() {}}),
    pagehide: () => listeners.pagehide(),
    request: () => JSON.parse(vm.runInContext('JSON.stringify(requestOptions())', context))
  };
}
test('CPU and GPU start without an additional browser reserve and remain configurable', () => {
  for (const device of ['cpu', 'cuda']) {
    const ui = studio(null, device);
    assert.equal(ui.control('bufferMs').value, '0');
    const immediate = ui.receiveAt([0, .04, .12]);
    assert.deepEqual(immediate.starts.map(x => Math.round(x * 1000)), [0, 80, 160]);
    assert.equal(immediate.underruns, 0);
    ui.control('bufferMs').value = '20';
    const buffered = ui.receiveAt([0, .04, .12]);
    assert.deepEqual(buffered.starts.map(x => Math.round(x * 1000)), [20, 100, 180]);
    assert.equal(buffered.underruns, 0);
    ui.reset();
    assert.equal(ui.control('bufferMs').value, '0');
  }
});
test('underrun metric includes the silence added when playback restarts', () => {
  const ui = studio();
  ui.control('bufferMs').value = '80';
  const result = ui.receiveAt([0, .20, .25]);
  assert.equal(result.underruns, 1);
  assert.ok(Math.abs(result.gapMs - 120) < 1e-8);
  assert.deepEqual(result.starts.map(x => Math.round(x * 1000)), [80, 280, 360]);
});

test('floating-point noise at a chunk boundary does not insert a new startup reserve', () => {
  const ui = studio();
  ui.control('bufferMs').value = '80';
  const result = ui.receiveAt([0, .16, .2400000000000001]);
  assert.equal(result.underruns, 0);
  assert.equal(result.gapMs, 0);
  assert.deepEqual(result.starts.map(x => Math.round(x * 1000)), [80, 160, 240]);
});

test('chunks arriving before the scheduled end stay contiguous', () => {
  const ui = studio();
  ui.control('bufferMs').value = '80';
  const result = ui.receiveAt([0, .10, .15]);
  assert.equal(result.underruns, 0);
  assert.equal(result.gapMs, 0);
  assert.deepEqual(result.starts.map(x => Math.round(x * 1000)), [80, 160, 240]);
});

test('null codec defaults stay omitted rather than becoming greedy false', () => {
  const ui = studio();
  const payload = ui.request();
  assert.equal(payload.do_sample, true);
  for (const field of ['subtalker_do_sample', 'subtalker_temperature', 'subtalker_top_p', 'subtalker_top_k'])
    assert.equal(Object.hasOwn(payload, field), false, field);
  ui.control('do_sample').checked = false;
  assert.equal(ui.request().do_sample, false);
  assert.equal(Object.hasOwn(ui.request(), 'subtalker_do_sample'), false);
});
for (const value of [true, false]) {
  test('explicit codec default ' + value + ' is preserved', () => {
    const ui = studio(value);
    assert.equal(ui.request().subtalker_do_sample, value);
  });
  test('user can explicitly choose codec sampling ' + value + ' then reset', () => {
    const ui = studio();
    ui.control('subtalker_do_sample').value = String(value);
    assert.equal(ui.request().subtalker_do_sample, value);
    ui.reset();
    assert.equal(Object.hasOwn(ui.request(), 'subtalker_do_sample'), false);
  });
}
test('invalid codec sampling value cannot silently disable sampling', () => {
  const ui = studio();
  ui.control('subtalker_do_sample').value = 'invalid';
  assert.throws(ui.request, /Codec Sampling/);
});

test('Connect primes one shared context and pagehide closes it', async () => {
  const ui = studio();
  ui.connectGesture();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(ui.context().resumeCalls, 1);
  assert.equal(ui.context().state, 'running');
  const connectReady = ui.events().filter(event => event.type === 'audio_ready' && event.phase === 'connect');
  assert.equal(connectReady.length, 1);
  ui.pagehide();
  assert.equal(ui.context().closeCalls, 1);
  assert.equal(ui.context().state, 'closed');
});
test('speech overlaps audio activation and stop cancels queued and scheduled sources', async () => {
  const ui = studio(null, 'cpu', {deferResume: true});
  const start = ui.start();
  const ws = ui.socket();
  ws.open();
  ws.message(new Int16Array(1920).buffer);
  assert.equal(ui.context().sources.length, 0);
  ui.context().resolveResume();
  await start;
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(ui.context().sources.length, 1);
  assert.ok(ui.events().some(event => event.type === 'ws_open'));
  assert.ok(ui.events().some(event => event.type === 'audio_ready'));
  const source = ui.context().sources[0];
  ui.stop();
  assert.equal(source.stopCalls, 1);
  assert.equal(source.disconnectCalls, 1);
  assert.equal(ui.context().closeCalls, 0);
  assert.equal(ui.control('speak').disabled, false);
});
test('finish waits for queued playback to drain before re-enabling speech', async () => {
  const ui = studio();
  const start = ui.start();
  const ws = ui.socket();
  ws.open();
  await start;
  await Promise.resolve();
  ws.message(new Int16Array(1920).buffer);
  await Promise.resolve();
  const source = ui.context().sources[0];
  ws.message(JSON.stringify({type: 'done'}));
  assert.equal(ui.control('speak').disabled, true);
  source.onended();
  assert.equal(ui.control('speak').disabled, false);
  assert.equal(ui.context().closeCalls, 0);
});
