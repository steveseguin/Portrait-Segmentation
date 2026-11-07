'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const tf = require('@tensorflow/tfjs');

// Evaluate the actual processing functions, without starting camera/model I/O.
const source = fs.readFileSync(path.join(__dirname, '..', 'index.js'), 'utf8');
const end = source.indexOf('/* Set up the on-start listener */');
assert.ok(end > 0, 'Processing functions must precede the UI listeners');
const context = vm.createContext({
  tf,
  document: { getElementById: () => ({}) },
});
vm.runInContext(source.slice(0, end), context);

function assertUnitRange(tensor) {
  for (const value of tensor.dataSync()) {
    assert.ok(Number.isFinite(value) && value >= 0 && value <= 1);
  }
}

test('an all-background prediction remains an empty mask', () => {
  const mask = tf.zeros([1, 128, 128, 1]);
  const result = context.refine(mask);
  try {
    assert.deepEqual(result.shape, [128, 128, 1]);
    assert.ok(Array.from(result.dataSync()).every(value => value === 0));
  } finally {
    tf.dispose([mask, result]);
  }
});

test('an empty prediction renders the selected background, not black', async () => {
  const image = tf.ones([128, 128, 3]);
  const mask = tf.zeros([1, 128, 128, 1]);
  const background = tf.fill([300, 300, 3], 0.4);
  context.background = background;
  vm.runInContext('bgim = background;', context);
  const result = context.process(image, mask);
  try {
    const expected = background.dataSync()[0];
    assert.ok(Array.from(result.dataSync()).every(value => value === expected));
    const pixels = await tf.browser.toPixels(result);
    for (let i = 0; i < pixels.length; i += 4) {
      assert.deepEqual(Array.from(pixels.slice(i, i + 4)), [102, 102, 102, 255]);
    }
  } finally {
    tf.dispose([image, mask, background, result]);
  }
});

test('a mixed prediction retains foreground and background regions', () => {
  const values = new Float32Array(128 * 128);
  for (let y = 0; y < 128; y++) values.fill(1, y * 128 + 64, (y + 1) * 128);
  const mask = tf.tensor4d(values, [1, 128, 128, 1]);
  const result = context.refine(mask);
  try {
    assertUnitRange(result);
    const data = result.dataSync();
    assert.equal(data[64 * 128 + 16], 0);
    assert.equal(data[64 * 128 + 96], 1);
  } finally {
    tf.dispose([mask, result]);
  }
});

test('an all-foreground prediction remains finite and keeps its interior', () => {
  const mask = tf.ones([1, 128, 128, 1]);
  const result = context.refine(mask);
  try {
    assertUnitRange(result);
    assert.equal(result.dataSync()[64 * 128 + 64], 1);
  } finally {
    tf.dispose([mask, result]);
  }
});

test('alternating empty and foreground frames do not retain tensors', () => {
  const before = tf.memory().numTensors;
  for (let i = 0; i < 10; i++) {
    const mask = tf.fill([1, 128, 128, 1], i % 2);
    const result = context.refine(mask);
    try {
      assertUnitRange(result);
    } finally {
      tf.dispose([mask, result]);
    }
  }
  assert.equal(tf.memory().numTensors, before);
});
