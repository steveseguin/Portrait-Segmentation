# Browser processing regression tests

From `tensorflowjs`, use Node.js 20 or newer and run:

```sh
npm ci --ignore-scripts
npm test
```

The tests use TensorFlow.js 1.2.8, matching `index.html`, on its CPU backend.
They evaluate the processing functions from `index.js` without starting the
camera or loading model weights. Empty, full and mixed foreground masks exercise
real tensor operations, blending and pixel conversion. These checks do not run
the browser UI, WebGL backend, camera or model inference.
