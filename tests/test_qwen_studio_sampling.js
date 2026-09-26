'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../RealtimeTTS/studio/studio.js'), 'utf8');

function studio(codecDefault = null, device = 'cpu') {
  const controls = new Map();
  const initial = {
    voice: 'reference', language: 'de', temperature: '0.9', top_p: '1',
    top_k: '50', seed: '-1', max_new_tokens: '2048', repetition_penalty: '1.05',
    subtalker_temperature: '', subtalker_top_p: '', subtalker_top_k: '',
    subtalker_do_sample: ''
  };
  const canvas = new Proxy({}, {get: () => () => {}});
  function control(id) {
    if (!controls.has(id)) {
      let value = initial[id] || '';
      controls.set(id, {
        get value() { return value; },
        set value(next) { value = next == null ? '' : String(next); },
        checked: true, disabled: id === 'instructions', clientWidth: 600,
        classList: {toggle() {}, add() {}, remove() {}},
        checkValidity: () => true, getContext: () => canvas,
        dispatchEvent() {}
      });
    }
    return controls.get(id);
  }
  const clone = {value: 'speaker_only', checked: true, dispatchEvent() {}};
  const context = vm.createContext({
    document: {getElementById: control, querySelectorAll: () => [], querySelector: () => clone},
    location: {origin: 'http://localhost'}, devicePixelRatio: 1, performance: {now: () => 0},
    window: {addEventListener() {}}, Event: class {},
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
            started: 0, audible: false, underruns: 0, gapMs: 0};
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
    request: () => JSON.parse(vm.runInContext('JSON.stringify(requestOptions())', context))
  };
}
test('CPU and GPU playback both default to 80 ms startup reserve', () => {
  assert.equal(studio().control('bufferMs').value, '80');
  assert.equal(studio(null, 'cuda').control('bufferMs').value, '80');
});
test('underrun metric includes the silence added when playback restarts', () => {
  const result = studio().receiveAt([0, .20, .25]);
  assert.equal(result.underruns, 1);
  assert.ok(Math.abs(result.gapMs - 120) < 1e-8);
  assert.deepEqual(result.starts.map(x => Math.round(x * 1000)), [80, 280, 360]);
});

test('floating-point noise at a chunk boundary does not insert a new startup reserve', () => {
  const result = studio().receiveAt([0, .16, .2400000000000001]);
  assert.equal(result.underruns, 0);
  assert.equal(result.gapMs, 0);
  assert.deepEqual(result.starts.map(x => Math.round(x * 1000)), [80, 160, 240]);
});

test('chunks arriving before the scheduled end stay contiguous', () => {
  const result = studio().receiveAt([0, .10, .15]);
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
