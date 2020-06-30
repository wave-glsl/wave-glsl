// http://localhost:2281/web/?n=512&s=64&t=8&a=3000
const USP = new URLSearchParams(location.search);
const USE_GPU = USP.get('gpu') != 0;
const ITERATIONS = USP.get('i') === null ? 7 : +USP.get('i');
const INITIAL_GRID_SIZE = +USP.get('n') || 128;
const INITIAL_AMP = param('a', 100);
const INITIAL_FREQ = +USP.get('f') || 20;
const SCHRODINGER = +USP.get('sch') || 0;
const MIN_GRID_SIZE = 64;
const MAX_GRID_SIZE = 2048;
const RESIZE_FACTOR = 2;
const WAVE_SPEED = +USP.get('c') || 1;
const NOISE_AMP = 1e-2;
const NOISE_FREQ = +USP.get('nf') || 0;
const NOISE_RADIUS = 0.5;
const DISH_SIZE = 1 + 5 / INITIAL_GRID_SIZE; // to reduce rounding error in dx=D/N
const EDGE_RADIUS = 1 / DISH_SIZE;
const EDGE_SHARPNESS = +USP.get('e') || 1 / INITIAL_GRID_SIZE;
const AMP_FACTOR = 1.05;
const FREQ_FACTOR = 2 ** (1 / 12); // en.wikipedia.org/wiki/Piano_key_frequencies
const FREQ_SCALE = +USP.get('fs') || 500;
const UPDATE_STEPS = +USP.get('s') || 64;
const DAMPING = +USP.get('g') || 0;
const MODULATION = +USP.get('m') || 1e-3;
const cos = Math.cos;
const sin = Math.sin;
const PI = Math.PI;
const INPUT_WAVE = parseInputWave(USP.get('f') || '7.77');
const WAVE_FILENAME = 'wave_surface.obj'
const RING_FILENAME = 'led_ring.obj';
const MAX_CANVAS_SIZE = 1024;
const DT_DIVIDER = +USP.get('dt') || 2;
const WAVE_AMP_MAX = 1e+3;
const WAVE_AMP_MIN = 1e-3;
const WAVE_AMP_NORMALIZED = 1;
const ANIMATE_WAVE_MESSAGE = 'wave-update';

const UPDATE_INTERVAL = 0;
const RENDER_INTERVAL = +USP.get('r') || 0; // ms, 0 for continuous rendering
const STATS_INTERVAL = 1000; // ms, 0 to disable
const STATS_ID = '#stats';

let WaveSolver = USE_GPU ? GpuWaveSolver : WasmWaveSolver;
let nUpdateSteps = UPDATE_STEPS;
let drivingAmp = INITIAL_AMP;
let running = false;
let drawIsolines = false;
let numSteps = 0;
let computeTime = 0;
let numRenderedFrames = 0;
let prevWaveTime = 0;
let prevRenderTime = -Infinity;
let prevStatsTime = -Infinity;
let pstats = document.querySelector(STATS_ID);
let canvas = document.querySelector('canvas');
let spectrogram, audioCanvas;
let wave = createWave();

window.onload = () => void main();

async function main() {
  while (!wave.init()) {
    console.log('waiting for wasm');
    await sleep(250);
  }

  setKeyboardHandlers();
  setMouseHandlers();
  setUpdateMessageHandler();

  wave.setInitialEdge(EDGE_RADIUS, EDGE_SHARPNESS);
  addWaveSplash();
  renderWave();
}

function setUpdateMessageHandler() {
  if (RENDER_INTERVAL > 0) {
    window.onmessage = e => {
      if (e.data === ANIMATE_WAVE_MESSAGE)
        animateWave();
    };
  }
}

function param(name, value) {
  let x = USP.get(name);
  return x === null ? value : x;
}

function setKeyboardHandlers() {
  console.log('s: start/stop');
  console.log('p: one step forward');
  console.log('m: microphone spectrogram recording');
  console.log('i: switch between rgba <-> isolines');
  console.log('u: upsize wave grid for better quality');
  console.log('d: downsize wave grid for faster simulation');
  console.log('c: decrease driving amp');
  console.log('v: increase driving amp');
  console.log('b: decrease speed');
  console.log('n: increase speed');
  console.log('w: make a wavefront obj file');
  console.log('a: normalize wave amplitude 3*sigma -> 1.0');
  console.log('f: upload an audio file');

  document.onkeypress = e => {
    switch (e.key) {
      case 'f':
        selectAudioFile();
        break;
      case 'a':
        console.log('Normalizing wave amplitude');
        wave.normalizeAmplitude();
        renderWave();
        printStats();
        break;
      case 'w':
        console.log('generating a wave surface obj file');
        downloadFile(
          WAVE_FILENAME,
          createWaveSurfaceMesh());
        break;
      case 'l':
        console.log('generating a led ring obj file');
        downloadFile(
          RING_FILENAME,
          createLedLightMesh());
        break;
      case 'b':
        nUpdateSteps = Math.max(1, nUpdateSteps - 1);
        console.log('update steps:', nUpdateSteps);
        break;
      case 'n':
        nUpdateSteps = Math.min(1024, nUpdateSteps + 1);
        console.log('update steps:', nUpdateSteps);
        break;
      case 'c':
        drivingAmp /= AMP_FACTOR;
        console.log('driving amp:', drivingAmp);
        break;
      case 'v':
        drivingAmp *= AMP_FACTOR;
        console.log('driving amp:', drivingAmp);
        break;
      case 'u':
        resizeWave(wave.gridSize * RESIZE_FACTOR | 0);
        renderWave();
        break;
      case 'd':
        resizeWave(wave.gridSize / RESIZE_FACTOR | 0);
        renderWave();
        break;
      case 's':
        if (audioCanvas) {
          document.body.removeChild(audioCanvas);
          audioCanvas = null;
          canvas.hidden = false;
          spectrogram.stop();
          console.log('Using recorded sound as input:',
            spectrogram.recordedSound.length *
            spectrogram.frameSize | 0, 'sec');
          // The max volume in WebAudio is about -30 dB = 0.03 m.
          // The max audible frequency is 10 kHz, which is about
          // the range of piano key frequencies; also most bird songs
          // fit into the 6 kHz range (www.fssbirding.org.uk). At the
          // same time, the NxN grid simulation can't possibly render
          // a cos(2pi*w*x) wave with w > N/2, as in the extreme case
          // a "wave" will be represented by alternating pixels +1/-1.
          // In practice, it can deal with w < N/sqrt(2), apparently
          // using the diagonal length, and beyond that the simulation
          // turns into noise.
          wave.inputWave = t => {
            let a = spectrogram.getInterpolatedAmp(t / FREQ_SCALE);
            return [
              a[0] * drivingAmp * 30 - DAMPING,
              a[1] * drivingAmp * 30];
          };
        }
        startStop();
        break;
      case 'p':
        wave.computeNextStep();
        renderWave();
        printStats();
        break;
      case 'i':
        drawIsolines = !drawIsolines;
        renderWave();
        break;
      case 'm':
        startAudioRecording();
        break;
    }
  };
}

function startAudioRecording(audioStream = null) {
  if (!spectrogram) {
    console.log('Creating a <canvas> for the spectrogram');
    canvas.hidden = true;
    audioCanvas = document.createElement('canvas');
    audioCanvas.width = MAX_CANVAS_SIZE;
    audioCanvas.height = MAX_CANVAS_SIZE;
    document.body.appendChild(audioCanvas);
    spectrogram = new Spectrogram(audioCanvas);
    spectrogram.start(audioStream);
  } else {
    console.warn('Spectrogram already running');
  }
}

async function selectAudioFile() {
  console.log('Creating an <input> to pick a file');
  let input = document.createElement('input');
  input.type = 'file';
  input.accept = 'audio/mpeg';
  input.click();

  let file = await new Promise((resolve, reject) => {
    input.onchange = () => {
      let files = input.files || [];
      if (files.length > 0)
        resolve(files[0]);
      else
        reject(new Error(`No files selected`));
    };
  });

  console.log('Selected file:', file.type, file.size, 'bytes');

  console.log('Creating an <audio> element to render the file');
  let audio = document.createElement('audio');
  audio.src = URL.createObjectURL(file);

  await new Promise(resolve => {
    audio.onloadeddata =
      () => resolve();
  });

  console.log('Audio loaded');
  audio.loop = true;
  audio.play();

  let stream = audio.captureStream();
  console.log('Got media stream from <audio>:', stream.id,
    'tracks:', stream.getTracks().map(t => t));

  startAudioRecording(stream);
}

function setMouseHandlers() {
  console.log(`click: start/stop`);
  document.body.onclick = e => {
    if (e.target === canvas) {
      if (!running) {
        let x = e.clientX / canvas.clientWidth * 2 - 1;
        let y = e.clientY / canvas.clientHeight * 2 - 1;
        addWaveSplash(x, y);
      }

      startStop();
    }
  };
}

function addWaveSplash(x = 0, y = 0) {
  wave.setInitialWave(
    NOISE_AMP, NOISE_FREQ, NOISE_RADIUS, { x, y });
}

function startStop() {
  running = !running;
  console.log(running ? 'started' : 'stopped');
  if (running) {
    animateWave();
  } else {
    wave.normalizeAmplitude();
    renderWave();
  }
}

function printStats(
  stats = wave.getWaveStats(),
  time = performance.now()) {

  let [avgx, avgy, rmax, rdev] = stats;

  let ns = numSteps;
  let ct = computeTime / 1e3;
  let nr = numRenderedFrames;
  let wt = wave.waveTime;
  let n2 = wave.gridSize ** 2;
  let dt = (time - prevStatsTime) / 1e3;
  let dwt = wt - prevWaveTime;
  let title = 'T+' + wt.toFixed(1) + 's';

  pstats.textContent = [
    title, (dwt / dt).toFixed(2) + 'x', '|',
    'sim', xtonum(ns / ct, 'fps'),
    xtonum(ns / ct * n2, 'pps'),
    (ct / dt * 100 | 0) + '%', '|',
    'amp', rmax.toExponential(1),
    'avg', ((avgx ** 2 + avgy ** 2) ** 0.5).toExponential(2), '|',
    'draw', nr / dt | 0, 'fps'
  ].join(' ');

  numSteps = 0;
  computeTime = 0;
  numRenderedFrames = 0;
  prevWaveTime = wt;

  document.title = title;
}

function xtonum(x, units) {
  if (x > 1e12) return (x / 1e12 | 0) + ' T' + units;
  if (x > 1e9) return (x / 1e9 | 0) + ' G' + units;
  if (x > 1e6) return (x / 1e6 | 0) + ' M' + units;
  if (x > 1e3) return (x / 1e3 | 0) + ' K' + units;
  return (x | 0) + ' ' + units;
}

function animateWave() {
  let prevTime = performance.now();

  for (let i = 0; i < nUpdateSteps; i++)
    wave.computeNextStep();

  let time = performance.now();
  computeTime += time - prevTime;
  numSteps += nUpdateSteps;

  if (time >= prevStatsTime + STATS_INTERVAL) {
    let [min, max, avg, stddev] = wave.getWaveStats()
    printStats([min, max, avg, stddev], time);
    prevStatsTime = time;

    if (stddev * 3 > WAVE_AMP_MAX || stddev * 3 < WAVE_AMP_MIN) {
      console.log('Rescaling wave amplitude from',
        (stddev * 3).toExponential(2), 'to', WAVE_AMP_NORMALIZED);
      wave.normalizeAmplitude(WAVE_AMP_NORMALIZED);
    }
  }

  if (time >= prevRenderTime + RENDER_INTERVAL) {
    renderWave();
    prevRenderTime = time;
    numRenderedFrames++;
  }

  if (running) {
    if (RENDER_INTERVAL > 0)
      setTimeout(animateWave, 0);
    else
      requestAnimationFrame(animateWave);
  }
}

function renderWave() {
  wave.renderWaveImage(drawIsolines);
}

function sleep(time_msec) {
  return new Promise(
    resolve => setTimeout(resolve, time_msec));
}

function resizeWave(newGridSize) {
  if (newGridSize < MIN_GRID_SIZE || newGridSize > MAX_GRID_SIZE) {
    console.warn(`can't resize wave to`, newGridSize);
    return false;
  }

  if (newGridSize != Math.round(newGridSize))
    throw new Error(`Invalid wave size: ${newGridSize}`);

  console.log('resizing wave from',
    wave.gridSize, 'to', newGridSize);

  let wave2 = createWave({
    gridSize: newGridSize,
    inputWave: wave.inputWave,
    waveTime: wave.waveTime,
  });

  wave2.init();

  let waveData = wave.getWaveData();
  let edgeData = wave.getEdgeData();
  let wget = (data, i, j) => data[i + j * wave.gridSize];

  let interpolate = (srcData, x, y) => {
    let i = x / wave.gridStep;
    let j = y / wave.gridStep;
    let i1 = Math.floor(i);
    let j1 = Math.floor(j);
    let i2 = Math.ceil(i) % wave.gridSize;
    let j2 = Math.ceil(j) % wave.gridSize;
    let xr = i - i1;
    let yr = j - j1;
    return 0 +
      wget(srcData, i1, j1) * (1 - xr) * (1 - yr) +
      wget(srcData, i2, j2) * xr * yr +
      wget(srcData, i2, j1) * xr * (1 - yr) +
      wget(srcData, i1, j2) * (1 - xr) * yr;
  };

  wave2.setWaveData((x, y) =>
    interpolate(waveData, x, y));

  wave2.setEdgeData((x, y) =>
    interpolate(edgeData, x, y));

  wave.delete();
  wave = wave2;
}

function createWave(args) {
  return new WaveSolver({
    canvas,
    pstats,
    gridSize: INITIAL_GRID_SIZE,
    dishSize: DISH_SIZE,
    waveSpeed: WAVE_SPEED,
    inputWave: INPUT_WAVE,
    iterations: ITERATIONS,
    timeStepDivider: DT_DIVIDER,
    maxCanvasSize: MAX_CANVAS_SIZE,
    schrodinger: SCHRODINGER,
    ...args
  });
}

function downloadFile(filename, text) {
  let blob = new Blob([text], { type: 'text/plain' });
  let url = URL.createObjectURL(blob);
  console.log(`Wavefront obj file: ${text.length / 1024 | 0} KB, ${url}`);
  let a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function createLedLightMesh(r1 = 0.5, r2 = 1.0, dh = 0.1, n = 60, prec = 3) {
  let f2s = x => x.toFixed(prec).replace(/\.?0+$/, '');
  let vertices = [];
  let triangles = [];

  for (let i = 0; i < n; i++) {
    let a = i / n * 2 * Math.PI;
    let x = Math.sin(a);
    let y = Math.cos(a);

    vertices[i + 0 * n] = [x * r1, y * r1, 0];
    vertices[i + 1 * n] = [x * r1, y * r1, dh];
    vertices[i + 2 * n] = [x * r2, y * r2, 0];
    vertices[i + 3 * n] = [x * r2, y * r2, dh];
  }

  for (let d = 1; d <= n; d++) {
    let b = d < n ? d + 1 : 1;
    let q = d + n * 2;
    let p = b + n * 2;

    // normals facing outwards, or the ccw direction
    triangles.push([b, d, q]);
    triangles.push([b, q, p]);
    triangles.push([d + n, b + n, q + n]);
    triangles.push([q + n, b + n, p + n]);
    triangles.push([p, q, q + n]);
    triangles.push([p, q + n, p + n]);
    triangles.push([d, b, d + n]);
    triangles.push([b, b + n, d + n]);
  }

  let text = [];

  text.push(`# Disk with ${n} segments`);
  text.push(`# ${vertices.length} vertices, ${triangles.length} triangles`);
  text.push(`# ${r1} < r < ${r2}, 0 < z < ${dh}`);

  text.push('');
  for (let v of vertices)
    text.push('v ' + v.map(f2s).join(' '));

  text.push('');
  for (let t of triangles)
    text.push('f ' + t.join(' '));

  return text.join('\n');
}

function createWaveSurfaceMesh() {
  let n = wave.gridSize;
  let prec = Math.ceil(Math.log10(n));
  let data = wave.getWaveData();
  let [min, max, avg, dev] = wave.getWaveStats();
  let text = [];

  text.push(`# ${n} x ${n} wave, ${n * n} vertices, ${2 * (n - 1) ** 2} triangles`);
  text.push(`# -1 < x, y, z < +1`);
  text.push('');

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let x = (0.5 + i) / n - 0.5;
      let y = (0.5 + j) / n - 0.5;
      let z = data[i + j * n];

      // clamp to the 3 sigma range
      z = (z - avg) / (300 * dev);
      if (z < -1) z = -1;
      if (z > +1) z = +1;

      let sx = x.toFixed(prec).replace(/\.?0+$/, '');
      let sy = y.toFixed(prec).replace(/\.?0+$/, '');
      let sz = z.toFixed(prec).replace(/\.?0+$/, '');

      text.push(`v ${sx} ${sy} ${sz}`);
    }
  }

  text.push('');

  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < n - 1; j++) {
      // In wavefront obj files, vertex indices
      // are 1-based. The polygon normals must
      // be consistently directed up.
      //
      //    a b
      //    c d
      //
      let a = i + j * n + 1;
      let b = a + 1;
      let d = b + n;
      let c = d - 1;

      text.push(`f ${a} ${c} ${b}`);
      text.push(`f ${c} ${d} ${b}`);
    }
  }

  return text.join('\n');
}

function parseInputWave(str) {
  if (/^(\d+(\.\d+)?,?)+$/.test(str)) {
    let freqs = str.split(',')
      .map(s => parseFloat(s) || 0);
    console.log('freqs:', freqs);
    return t => {
      let re = 0;
      let im = 0;

      for (let freq of freqs) {
        re += cos(freq * t);
        im += sin(freq * t);
      }

      return [
        re * drivingAmp - DAMPING,
        im * drivingAmp,
      ];
    };
  }

  console.error('Invalid freqs spec: ?f=' + freqs);
  return t => -DAMPING;
}
