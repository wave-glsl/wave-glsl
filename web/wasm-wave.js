class WasmWaveSolver {
  constructor(args) {
    this.canvas = args.canvas;
    this.pstats = args.pstats;
    this.gridSize = args.gridSize | 0;
    this.dishSize = args.dishSize;
    this.waveSpeed = args.waveSpeed;
    this.inputWave = args.inputWave;
    this.gridStep = this.dishSize / (this.gridSize - 1);
    this.timeStep = this.gridStep / this.waveSpeed / 20;
    this.waveTime = args.waveTime || 0;

    this.canvas.width = this.gridSize;
    this.canvas.height = this.gridSize;
    this.context2d = this.canvas.getContext('2d');
    this.imageData = this.context2d.getImageData(0, 0, this.gridSize, this.gridSize);

    this.printConfig();
  }

  init() {
    if (!Module.WaveSolver)
      return false;

    this.wasm = new Module.WaveSolver(
      this.gridSize,
      this.gridStep,
      this.timeStep,
      this.waveSpeed);

    this.wasm.init();
    this.setInitialWave();
    console.log('wave ready:', this.gridSize);
    return true;
  }

  delete() {
    this.wasm.delete();
    this.wasm = null;
  }

  getWaveData() {
    return new Float32Array(
      Module.HEAPF32.buffer,
      this.wasm.wave(),
      this.gridSize ** 2);
  }

  getEdgeData() {
    return new Float32Array(
      Module.HEAPF32.buffer,
      this.wasm.edge(),
      this.gridSize ** 2);
  }

  setWaveData(fn, data = this.getWaveData()) {
    let n = this.gridSize;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let x = i * this.gridStep;
        let y = j * this.gridStep;
        data[i + j * n] = fn(x, y);
      }
    }

    this.wasm.use_init_wave();
  }

  setEdgeData(fn) {
    this.setWaveData(fn, this.getEdgeData());
  }

  getWaveStats() {
    let data = this.getWaveData();
    let n2 = this.gridSize ** 2;

    let min = data[0];
    let max = data[0];
    let avg = data[0];
    let sd2 = 0;

    for (let i = 0; i < n2; i++) {
      min = Math.min(min, data[i]);
      max = Math.max(max, data[i]);
      avg += data[i] / n2;
    }

    for (let i = 0; i < n2; i++)
      sd2 += (data[i] - avg) ** 2 / n2;

    return [min, max, avg, sd2 ** 0.5];
  }

  setInitialWave(noiseAmp, noiseFreq = 0, noiseRadius = 0.5) {
    this.setWaveData((x, y) => {
      // return noiseAmp * (Math.random() * 2 - 1);
      x = x / this.dishSize * 2 - 1;
      y = y / this.dishSize * 2 - 1;
      let r = Math.sqrt(x ** 2 + y ** 2);
      let a = !r ? 0 : Math.acos(x / r) * (y >= 0 ? 1 : -1);
      let z = Math.exp(-1e4 * (r - noiseRadius) ** 2);
      return noiseAmp * z * Math.cos(a * noiseFreq);
    });
  }

  setInitialEdge(edgeRadius = 0.95) {
    if (!edgeRadius) return;

    this.setEdgeData((x, y) => {
      x = x / this.dishSize * 2 - 1;
      y = y / this.dishSize * 2 - 1;
      let r = Math.sqrt(x ** 2 + y ** 2);
      let d = Math.abs(r - edgeRadius);
      let z = Math.exp(-1e4 * d ** 2);
      return d < 0.01 ? 1 :
        d < 0.02 ? z : 0;
    });
  }

  computeNextStep() {
    let input = this.getInputAmp();
    this.wasm.next(input);
    this.waveTime += this.timeStep;
  }

  renderWaveImage(drawIsolines = false) {
    let wasmPtr = drawIsolines ?
      this.wasm.isoline() :
      this.wasm.rgba();
    let rgba = new Uint8ClampedArray(
      Module.HEAPU8.buffer,
      wasmPtr,
      4 * this.gridSize ** 2);
    this.imageData.data.set(rgba, 0);
    this.context2d.putImageData(this.imageData, 0, 0);
  }

  getInputAmp() {
    return this.inputWave(this.waveTime * 2 * Math.PI);
  }

  printConfig() {
    let dx = this.gridStep.toExponential(2);
    let dt = this.timeStep.toExponential(2);
    let hz = 1 / this.timeStep | 0;
    let ws = this.waveSpeed.toFixed(2);
    console.log(`wave: dx=${dx} dt=${dt}/${hz} Hz ws=${ws}`);
  }
}