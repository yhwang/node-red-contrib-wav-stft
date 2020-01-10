import wav = require('node-wav');
import { downsampleTo16K } from './audiowav16';
import * as tf from '@tensorflow/tfjs-node';

const SAMPLE_RATE = 16000;
const WINDOW_SIZE = 20; // 20 mS
const STRIDE_SIZE = 10; // 10 mS

/**
 * Represent Node-Red's runtime
 */
declare class NodeRed {
  nodes: NodeRedNodes;
}

declare class NodeRedWire {
  [index: number]: string;
}

declare class NodeRedWires {
  [index: number]: NodeRedWire;
}

/**
 * Represent Node-Red's configuration for a custom node
 * For this case, it's the configuration for DeepSpeech
 * custom node
 */
declare class NodeRedProperties {
  id: string;
  type: string;
  name: string;
  modelURL: string;
  wires: NodeRedWires;
}

/**
 * Represent Node-Red's nodes
 */
declare class NodeRedNodes {
  // tslint:disable-next-line:no-any
  createNode(node: any, props: NodeRedProperties): void;
  // tslint:disable-next-line:no-any
  registerType(type: string, ctor: any): void;
}

/**
 * Represent Node-Red's message that passes to a node
 */
declare class NodeRedMessage {
  payload: Buffer|string|tf.Tensor;
}

declare interface StatusOption {
  fill: 'red' | 'green' | 'yellow' | 'blue' | 'grey';
  shape: 'ring' | 'dot';
  text: string;
}

/**
 * Based on the window size and stride size to compose a
 * Float32Array
 * @param buff
 * @param window
 * @param stride
 */
function _getStrideBuff(buff: Float32Array, window: number, stride: number)
    : Float32Array {
  const shape = [ window, (buff.length - window)/stride + 1];
  const rev = new Float32Array(shape[0] * shape[1]);
  for( let i = 0; i < shape[1]; i++) {
    const tmp = new Float32Array(buff.buffer, (i * stride) * 4, window);
    rev.set(tmp, i * window);
  }
  return rev;
}

/**
 * Compute the spectrograms for the input samples(waveforms).
 * https://en.wikipedia.org/wiki/Short-time_Fourier_transform.
 * @param buff
 */
function stft(buff: Float32Array): tf.Tensor {
  const windowSize = SAMPLE_RATE * 0.001 * WINDOW_SIZE;
  const strideSize = SAMPLE_RATE * 0.001 * STRIDE_SIZE;
  const trancate = (buff.length - windowSize ) % strideSize;
  const trancatedBuff = new Float32Array(
      buff.buffer, 0, buff.length - trancate);
  const stridedBuff =
      _getStrideBuff(trancatedBuff, windowSize, strideSize);
  const shape =
      [(trancatedBuff.length - windowSize)/strideSize + 1, windowSize];
  return tf.tidy(() => {
    const hannWin = tf.hannWindow(windowSize).expandDims(0);
    const windows = tf.tensor(stridedBuff, shape, 'float32');
    // weighting
    let fft = windows.mul(hannWin).rfft().abs().square();
    // scaling
    const scale = hannWin.square().sum().mul(tf.scalar(SAMPLE_RATE));
    const fftMiddle = fft.stridedSlice([0,1], [shape[0], -1], [1, 1])
        .mul(tf.scalar(2.0, 'float32').div(scale));
    const fftFirstColumn = fft.stridedSlice([0, 0], [shape[0], 1], [1, 1])
        .div(scale);
    const fftLastColumn = fft.stridedSlice([0, -1], shape, [1, 1]).div(scale);
    fft = tf.concat([fftFirstColumn, fftMiddle, fftLastColumn], 1)
        .add(tf.scalar(1e-14)).log();
    // normalize
    const {mean, variance} = tf.moments(fft, 0, true);
    return fft.sub(mean).div(variance.sqrt().sub(tf.scalar(1e-6)))
        .expandDims(2).expandDims();
  });
}

// Module for a Node-Red custom node
export = function wav2STFT(RED: NodeRed) {

  class Wav2STFT {
    // tslint:disable-next-line:no-any
    on: (event: string, fn: (msg: any) => void) => void;
    send: (msg: NodeRedMessage) => void;
    status: (option: StatusOption) => void;
    log: (msg: string) => void;

    id: string;
    type: string;
    name: string;
    wires: NodeRedWires;

    constructor(config: NodeRedProperties) {
      this.id = config.id;
      this.type = config.type;
      this.name = config.name;
      this.wires = config.wires;

      RED.nodes.createNode(this, config);
      this.on('input', (msg: NodeRedMessage) => {
        this.handleRequest(msg);
      });

      this.on('close', (done: () => void) => {
        this.handleClose(done);
      });

    }

    // handle a single request
    handleRequest(msg: NodeRedMessage) {

      const result = wav.decode(msg.payload);
      let audio: Float32Array = result.channelData;
      if (result.sampleRate > SAMPLE_RATE) {
        this.log(
          `downsampling from ${result.sampleRate} to ${SAMPLE_RATE}`);
        audio = downsampleTo16K(result.channelData[0], result.sampleRate);
      }

      this.send({payload: stft(audio)});
    }

    handleClose(done: () => void) {
      // node level clean up
      done();
    }
  }

  RED.nodes.registerType('wav-stft', Wav2STFT);
};
