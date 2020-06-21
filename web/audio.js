const RGB = [
  [255, 255, 255],
  [255, 0, 0],
  [0, 255, 0],
  [0, 0, 255],
  [0, 0, 0],
];

class Spectrogram {
  constructor(canvas, fftSize = 2048, maxTimeSteps = 1024) {
    this.canvas = canvas;
    this.audioCtx = new AudioContext();
    this.analyser = this.audioCtx.createAnalyser();

    this.analyser.fftSize = fftSize;
    this.sampleRate = this.audioCtx.sampleRate;
    this.frameSize = this.analyser.frequencyBinCount
      / this.sampleRate;
    this.maxAmplitude = 10 ** (this.analyser.maxDecibels / 20);
    this.minAmplitude = 10 ** (this.analyser.minDecibels / 20);

    this.dataArray = new Uint8Array(
      this.analyser.frequencyBinCount);

    this.maxTimeSteps = maxTimeSteps;
    this.recordedSound = [];

    // this.canvas.width = timeSteps;
    // this.canvas.height = this.dataArray.length;

    this.canvasCtx = this.canvas.getContext("2d");
    this.imageData = this.canvasCtx.getImageData(
      0, 0, this.canvas.width, this.canvas.height);

    this.timeStep = 0;
    this.stats();
  }

  stats() {
    let a = this.analyser;
    let c = this.audioCtx;
    console.log('fft size:', a.fftSize);
    console.log('sample rate:', c.sampleRate, 'Hz');
    console.log('captured freq range:', c.sampleRate / 2, 'Hz');
    console.log('frame size:', 1e3 * this.frameSize | 0, 'ms');
    console.log('recorded range:',
      this.frameSize * this.maxTimeSteps | 0, 's',
      this.dataArray.length * this.maxTimeSteps / 1024 | 0, 'KB');
    let [dbmin, dbmax] = [a.minDecibels, a.maxDecibels];
    console.log('decibel range:', dbmin, '..', dbmax, 'dB');
    let db = n => (10 ** ((n / 255 * (dbmax - dbmin) + dbmin) / 20)).toExponential(2);
    console.log('amplitude colors:',
      '\n\twhite:', db(256),
      '\n\tred:', db(192),
      '\n\tgreen:', db(128),
      '\n\tblue:', db(64),
      '\n\tblack:', db(0));
  }

  async stop() {
    console.log('disconnecting audio stream:', this.stream.id);
    this.source.disconnect();
    for (let t of this.stream.getTracks())
      t.stop();
    this.stream = null;
    this.source = null;
  }

  async start(audioStream = null) {
    this.stream = audioStream ||
      await navigator.mediaDevices.getUserMedia({ audio: true });
    this.source = this.audioCtx.createMediaStreamSource(this.stream);
    this.source.connect(this.analyser);
    console.log('connected audio stream:', this.stream.id);

    let animate = () => {
      if (this.stream) {
        this.draw();
        requestAnimationFrame(animate);
      }
    };

    animate();
  }

  draw() {
    let w = this.canvas.width;
    let h = this.canvas.height;
    let t = this.timeStep % this.maxTimeSteps;
    let d = this.dataArray;
    let tmax = this.maxTimeSteps;
    let dmax = d.length;
    let fmax = this.sampleRate / 2;
    let step = 2e3 / fmax * dmax | 0; // blue line at every 2 kHz
    let rgba = this.imageData.data;

    // this.analyser.getByteTimeDomainData(this.dataArray);
    this.analyser.getByteFrequencyData(this.dataArray);
    this.recordedSound[t] = this.dataArray.slice(0);

    for (let i = 0; i < dmax; i++) {
      let ds = Math.min(i % step, step - i % step) / dmax;
      let x = t / tmax * w | 0;
      let y = h - 1 - (i / dmax * h | 0);
      let p = 4 * (y * w + x);
      let [r, g, b] = this.getInterpolatedColor(1 - d[i] / 255);
      rgba[p + 0] = r;
      rgba[p + 1] = g;
      rgba[p + 2] = Math.max(b, Math.exp(-3e5 * ds ** 2) * 255);
      rgba[p + 3] = i / dmax * fmax > 10e3 ? 200 : 255;
    }

    this.canvasCtx.putImageData(this.imageData, 0, 0);
    this.timeStep++;
  }

  getInterpolatedColor(x) {
    x = Math.max(0, Math.min(1, x));

    let r = RGB;
    let n = r.length;
    let i = x * (n - 1) | 0;
    let j = Math.min(i + 1, n - 1);
    let s = x * (n - 1) - i;
    let a = r[i];
    let b = r[j];

    return [
      a[0] * (1 - s) + b[0] * s,
      a[1] * (1 - s) + b[1] * s,
      a[2] * (1 - s) + b[2] * s,
    ];
  }

  getInterpolatedAmp(t) {
    let rs = this.recordedSound;
    if (!rs.length) return 0;
    let dbMin = this.analyser.minDecibels;
    let dbMax = this.analyser.maxDecibels;
    let ti = t / this.frameSize % rs.length | 0;
    let rst = rs[ti];
    let nfreq = rst.length; // = frequencyBinCount
    let dfreq = this.sampleRate / 2 / nfreq;
    let sum = 0;

    for (let fi = 1; fi < nfreq; fi++) {
      if (!rst[fi]) continue;
      let freq = fi * dfreq;
      let db = rst[fi] / 256 * (dbMax - dbMin) + dbMin;
      let energy = 10 ** (db / 10);
      let amp = energy ** 0.5;
      sum += amp * Math.cos(t * freq);
    }

    return sum;
  }

  getStats(a) {
    let n = a.length;
    let min = a[0];
    let max = a[0];
    let avg = 0;
    let sd2 = 0;

    for (let i = 0; i < n; i++) {
      min = Math.min(min, a[i]);
      max = Math.max(max, a[i]);
      avg += a[i] / n;
    }

    for (let i = 0; i < n; i++)
      sd2 += (a[i] - avg) ** 2 / (n - 1);

    return [min, max, avg, sd2 ** 0.5];
  }
}
