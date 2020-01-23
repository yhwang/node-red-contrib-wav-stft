# node-red-contrib-wav-stft
A Node-RED node to decode audio in WAV format and perform short-term Fourier
transform (STFT). It computes the STFT based on the `Window Size` and
`Stride Size` settings. The audio WAV should come in as Buffer data type
from previous node. Here are the processing procedures:
- Slice the whole audio into fix duration segments according to the
  `Window Size` value. The `Stride Size` decides the interval between
  adjacent segments.
- Compute the STFT and frequency bins is 161
- Insert a size 1 rank as the first rank and last rank

## Installation

### Prerequisite
For Tensorflow.js on Node.js
([@tensorflow/tfjs-node](https://www.npmjs.com/package/@tensorflow/tfjs-node)
or
[@tensorflow/tfjs-node-gpu](https://www.npmjs.com/package/@tensorflow/tfjs-node-gpu)),
it depends on Tensorflow shared libraries. Putting Tensorflow.js as the
dependency of custom Node-RED node may run into a situation that multiple
custom nodes install multiple `tfjs-node` module as their dependencies. While
loading multiple Tensorflow shared libraries in the same process, the process
would abord by hitting protobuf assertion.

Therefore, this module put `@tensorflow/tfjs-node` as peer dependency. You need
to install it with the Node-RED manully.

Install `@tensorflow/tfjs-node`:
```
npm install @tensorflow/tfjs-node
```


### Install this module:
Once you install the peer dependency, you can install this module:
```
npm install node-red-contrib-wav-stft
```