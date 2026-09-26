'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../RealtimeTTS/studio/studio.js'), 'utf8');

function studio(codecDefault = null) {
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
    location: {origin: 'http://localhost'}, devicePixelRatio: 1,
    window: {addEventListener() {}}, Event: class {},
    fetch: async () => ({ok: false})
  });
  vm.runInContext(source, context);
  const caps = {
    model: {id: 'qwen'}, engine: {clone_mode: 'speaker_only'},
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
    reset: () => vm.runInContext('defaults()', context),
    request: () => JSON.parse(vm.runInContext('JSON.stringify(requestOptions())', context))
  };
}
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
