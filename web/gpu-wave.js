class GpuWaveSolver {
  constructor(args) {
    this.canvas = args.canvas;
    this.gpuContext = new GpuContext(this.canvas);
    this.maxCanvasSize = args.maxCanvasSize || 1024;
    this.inputWave = args.inputWave || (t => 0);
    this.gridSize = args.gridSize;
    this.dishSize = args.dishSize || 1;
    this.waveSpeed = args.waveSpeed || 1;
    this.gridStep = this.dishSize / this.gridSize;
    this.timeStep = this.gridStep / this.waveSpeed / 2 / args.timeStepDivider;
    this.waveTime = args.waveTime || 0;
    this.iterations = args.iterations;
    this.schrodinger = args.schrodinger;

    if (this.schrodinger)
      this.timeStep = this.gridStep ** 2 / args.timeStepDivider;
  }

  init() {
    this.printConfig();
    canvas.width = Math.min(this.maxCanvasSize, this.gridSize);
    canvas.height = Math.min(this.maxCanvasSize, this.gridSize);
    this.gpuContext.init();
    this.tempWave = this.gpuContext.createFrameBuffer(this.gridSize, 2);
    this.gpSplash = new GpuSplashProgram(this.gpuContext.gl);
    this.gpBoundary = new GpuBoundaryProgram(this.gpuContext, this.gridSize);

    const ctor = this.schrodinger ? GpuSWaveProgram :
      this.iterations > 0 ? GpuWaveImplicitProgram :
        GpuWaveProgram;

    this.gpWave = new ctor(this.gpuContext, {
      iterations: this.iterations,
      boundary: this.gpBoundary.output,
      size: this.gridSize,
      speed: this.waveSpeed,
      dx: this.gridStep,
      dt: this.timeStep,
    });

    this.gpRescale = new GpuRescalingProgram(this.gpuContext);

    this.gpAmpStats = new GpuStatsProgram(this.gpuContext, {
      size: this.gridSize,
      dx: this.gridStep,
    });

    this.gpDisplay = new GpuDisplayProgram(this.gpuContext.gl, {
      stats: this.gpAmpStats.output,
      size: this.gridSize,
      dx: this.gridStep,
    });

    return true;
  }

  getWaveData() {
    this.computeNextStep();
    return this.gpWave.read();
  }

  getWaveStats() {
    this.gpAmpStats.run(this.gpWave.output);
    let [avgx, avgy, rmax, m2] = this.gpAmpStats.read();
    return [avgx, avgy, rmax, m2 ** 0.5 / this.gridSize];
  }

  normalizeAmplitude(maxamp = 1e-3) {
    let [avgx, avgy, rmax, stddev] = this.getWaveStats();

    if (!Number.isFinite(stddev))
      return;

    for (let frame of this.gpWave.frames) {
      this.gpRescale.run({
        output: this.tempWave,
        input: frame,
        shift: [-avgx, -avgy],
        scale: maxamp / (3 * stddev),
      });

      // Same as copying.
      this.gpRescale.run({
        output: frame,
        input: this.tempWave,
      });
    }
  }

  renderWaveImage() {
    this.gpAmpStats.run(
      this.gpWave.output);

    this.gpDisplay.run(
      this.gpWave.output);
  }

  setInitialWave(amp, freq, radius, { x = 0, y = 0 } = {}) {
    console.log(`Adding a spash at`, x.toFixed(2), y.toFixed(2));

    let gl = this.gpuContext.gl;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);

    for (let frame of this.gpWave.frames) {
      this.gpSplash.run({
        output: frame,
        x, y,
        radius,
        amplitude: amp,
        angularFreq: freq,
      });
    }

    gl.disable(gl.BLEND);
  }

  setInitialEdge(radius, sharpness = 0.01, thickness = 1e-4) {
    this.gpBoundary.run({
      radius,
      sharpness,
      thickness,
    });
  }

  computeNextStep() {
    this.gpWave.run(
      this.inputWave(
        this.waveTime * 2 * Math.PI));

    this.waveTime += this.timeStep;
  }

  printConfig() {
    let dx = this.gridStep.toExponential(2);
    let dt = this.timeStep.toExponential(2);
    let hz = 1 / this.timeStep | 0;
    let n = this.gridSize;
    console.log(`gpu wave: ${n} dx=${dx} dt=${dt} ${hz} Hz`);
  }
}

class GpuContext {
  constructor(canvas) {
    this.canvas = canvas;
    this.ext = null;
    this.gl = null;
  }

  createVertexShader(source) {
    return GpuProgram.createVertexShader(this.gl, source);
  }

  createFragmentShader(source) {
    return GpuProgram.createFragmentShader(this.gl, source);
  }

  createProgram(vertexShader, fragmentShader) {
    return new GpuProgram(
      this.gl,
      vertexShader,
      fragmentShader)
  }

  createFrameBuffer(size, components = 1) {
    if (size & (size - 1))
      throw new Error(`${size} != 2**n`);

    let gl = this.gl;
    let type = this.ext.floatTexType;
    let fmt =
      components == 1 ? this.ext.formatR :
        components == 2 ? this.ext.formatRG :
          this.ext.formatRGBA;

    gl.activeTexture(gl.TEXTURE0);
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, fmt.internalFormat, size, size, 0, fmt.format, type, null);

    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, size, size);
    gl.clear(gl.COLOR_BUFFER_BIT);

    return {
      texture,
      fbo,
      fmt,
      width: size,
      height: size,
      attach(id) {
        gl.activeTexture(gl.TEXTURE0 + id);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        return id;
      }
    };
  }

  init() {
    let canvas = this.canvas;

    console.log('initializing webgl context',
      canvas.width, 'x', canvas.height);

    let params = {
      alpha: true,
      depth: false,
      stencil: false,
      antialias: false,
      preserveDrawingBuffer: false,
    };

    let gl = canvas.getContext('webgl2', params);
    let isWebGL2 = !!gl;

    if (!isWebGL2) {
      gl = canvas.getContext('webgl', params) ||
        canvas.getContext('experimental-webgl', params);
    }

    if (!gl) {
      console.warn('getContext("webgl") blocked by getContext("2d")?');
      throw new Error('Cannot get WebGL context');
    }

    if (isWebGL2)
      gl.getExtension('EXT_color_buffer_float');
    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    let floatTexType = isWebGL2 ?
      gl.FLOAT :
      gl.getExtension('OES_texture_float').FLOAT_OES;

    let formatRGBA;
    let formatRG;
    let formatR;

    if (isWebGL2) {
      formatRGBA = this.getSupportedFormat(gl, gl.RGBA32F, gl.RGBA, floatTexType);
      formatRG = this.getSupportedFormat(gl, gl.RG32F, gl.RG, floatTexType);
      formatR = this.getSupportedFormat(gl, gl.R32F, gl.RED, floatTexType);
    } else {
      formatRGBA = this.getSupportedFormat(gl, gl.RGBA, gl.RGBA, floatTexType);
      formatRG = this.getSupportedFormat(gl, gl.RGBA, gl.RGBA, floatTexType);
      formatR = this.getSupportedFormat(gl, gl.RGBA, gl.RGBA, floatTexType);
    }

    this.gl = gl;

    this.ext = {
      formatRGBA,
      formatRG,
      formatR,
      floatTexType,
    };

    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
  }

  getSupportedFormat(gl, internalFormat, format, type) {
    if (!this.supportRenderTextureFormat(gl, internalFormat, format, type)) {
      switch (internalFormat) {
        case gl.R32F:
          return this.getSupportedFormat(gl, gl.RG32F, gl.RG, type);
        case gl.RG32F:
          return this.getSupportedFormat(gl, gl.RGBA32F, gl.RGBA, type);
        default:
          return null;
      }
    }

    return {
      internalFormat,
      format
    }
  }

  supportRenderTextureFormat(gl, internalFormat, format, type) {
    let texture = gl.createTexture();

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null);

    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    return status == gl.FRAMEBUFFER_COMPLETE;
  }
}

class GpuProgram {
  constructor(gl, vertexShader, fragmentShader) {
    this.gl = gl;
    this.uniforms = {};
    this.program = GpuProgram.createProgram(gl, vertexShader, fragmentShader);
    this.uniforms = this.getUniforms();
  }

  bind() {
    this.gl.useProgram(this.program);
  }

  getUniforms() {
    let gl = this.gl;
    let program = this.program;
    let uniforms = [];
    let uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < uniformCount; i++) {
      let uniform = gl.getActiveUniform(program, i);
      uniforms[uniform.name] = gl.getUniformLocation(program, uniform.name);
    }
    return uniforms;
  }

  static blit(gl, output) {
    let w = output ? output.width : gl.drawingBufferWidth;
    let h = output ? output.height : gl.drawingBufferHeight;
    gl.viewport(0, 0, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, output ? output.fbo : null);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }

  static createProgram(gl, vertexShader, fragmentShader) {
    let program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
      throw gl.getProgramInfoLog(program);

    return program;
  }

  static createFragmentShader(gl, source) {
    return GpuProgram.createShader(gl, gl.FRAGMENT_SHADER, source);
  }

  static createVertexShader(gl, source) {
    return GpuProgram.createShader(gl, gl.VERTEX_SHADER, source);
  }

  static createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
      throw gl.getShaderInfoLog(shader);

    return shader;
  }
}

class GpuSingleVertexShader {
  static get(gl) {
    return GpuProgram.createVertexShader(gl, `
      precision highp float;
      attribute vec2 aPosition;
      varying vec2 vUv;

      void main () {
          vUv = aPosition * 0.5 + 0.5;
          gl_Position = vec4(aPosition, 0.0, 1.0);
      }
    `);
  }
}

class GpuLaplaceVertexShader {
  static get(gl) {
    return GpuProgram.createVertexShader(gl, `
        precision highp float;
        attribute vec2 aPosition;

        varying vec2 vUv;

        varying vec2 vL;
        varying vec2 vR;
        varying vec2 vT;
        varying vec2 vB;

        varying vec2 vLT;
        varying vec2 vRT;
        varying vec2 vLB;
        varying vec2 vRB;

        uniform float dx;

        void main () {
            vUv = aPosition * 0.5 + 0.5;

            vL = vUv - vec2(dx, 0.0);
            vR = vUv + vec2(dx, 0.0);
            vT = vUv + vec2(0.0, dx);
            vB = vUv - vec2(0.0, dx);

            vLT = vUv + vec2(-dx, +dx);
            vRT = vUv + vec2(+dx, +dx);
            vRB = vUv + vec2(+dx, -dx);
            vLB = vUv + vec2(-dx, -dx);

            gl_Position = vec4(aPosition, 0.0, 1.0);
        }
    `);
  }
}

class GpuWaveProgram {
  get output() {
    return this.frames[0];
  }

  static utils = `
    // en.wikipedia.org/wiki/Discrete_Laplace_operator
    vec2 diffuse(sampler2D uPrev1) {
      vec2 u1 = texture2D(uPrev1, vUv).xy;

      vec2 uL = texture2D(uPrev1, vL).xy;
      vec2 uR = texture2D(uPrev1, vR).xy;
      vec2 uT = texture2D(uPrev1, vT).xy;
      vec2 uB = texture2D(uPrev1, vB).xy;

      vec2 uLT = texture2D(uPrev1, vLT).xy;
      vec2 uRT = texture2D(uPrev1, vRT).xy;
      vec2 uLB = texture2D(uPrev1, vLB).xy;
      vec2 uRB = texture2D(uPrev1, vRB).xy;
      
      return
        + 0.25 * (uLT - u1)
        + 0.25 * (uRT - u1)
        + 0.25 * (uLB - u1)
        + 0.25 * (uRB - u1)
        + 0.50 * (uL - u1)
        + 0.50 * (uR - u1)
        + 0.50 * (uT - u1)
        + 0.50 * (uB - u1);
    }

    vec2 xmul(vec2 a, vec2 b) {
      return vec2(
        a.x*b.x - a.y*b.y,
        a.x*b.y + a.y*b.x);
    }

    vec2 xinv(vec2 a) {
      float r2 = dot(a, a);
      return vec2(a.x / r2, -a.y / r2);
    }

    vec2 xdiv(vec2 a, vec2 b) {
      return xmul(a, xinv(b));
    }
  `;

  constructor(gpuContext, { boundary, size, speed, dx, dt }) {
    console.log('Explicit wave solver:', size, 'dt', dt);
    let gl = gpuContext.gl;

    this.gl = gl;
    this.size = size;
    this.dx = dx;
    this.boundary = boundary;

    this.frames = [
      gpuContext.createFrameBuffer(size, 2),
      gpuContext.createFrameBuffer(size, 2),
      gpuContext.createFrameBuffer(size, 2),
    ];

    this.vertexShader = GpuLaplaceVertexShader.get(gl);

    this.fragmentShader = GpuProgram.createFragmentShader(gl, `
        precision highp float;
        precision highp sampler2D;

        varying vec2 vUv;

        varying vec2 vL;
        varying vec2 vR;
        varying vec2 vT;
        varying vec2 vB;

        varying vec2 vLT;
        varying vec2 vRT;
        varying vec2 vLB;
        varying vec2 vRB;        

        uniform sampler2D uBoundary;
        uniform sampler2D uPrev1;
        uniform sampler2D uPrev2;
        uniform vec2 inputAmp;
        uniform float dt;

        ${GpuWaveProgram.utils}

        void main () {
            float c2 = ${(speed * dt / dx) ** 2};
            vec2 g2 = ${0.5 * dt}*inputAmp;
            float b = texture2D(uBoundary, vUv).r;

            vec2 u2 = texture2D(uPrev2, vUv).xy;
            vec2 u1 = texture2D(uPrev1, vUv).xy;

            vec2 diff = diffuse(uPrev1);

            vec2 u0 = u1 + (u1 - u2) - xmul(g2, u2) + c2*diff;
            u0 = xmul(u0 * (1.0 - b), xinv(vec2(1.0, 0.0) - g2));
            
            gl_FragColor = vec4(u0, 0.0, 0.0);
        }
    `);

    this.gp = new GpuProgram(
      this.gl,
      this.vertexShader,
      this.fragmentShader);
  }

  run(force) {
    let [f0, f1, f2] = this.frames;
    this.frames = [f2, f0, f1];

    let gl = this.gl;
    let gp = this.gp;
    let uf = gp.uniforms;
    gp.bind();
    gl.uniform2f(uf.inputAmp, force[0], force[1]);
    gl.uniform1f(uf.dx, this.dx);
    gl.uniform1i(uf.uBoundary, this.boundary.attach(0));
    gl.uniform1i(uf.uPrev1, f0.attach(1));
    gl.uniform1i(uf.uPrev2, f1.attach(2));
    GpuProgram.blit(gl, f2);
  }

  read() {
    let n = this.size;
    let rgba = new Float32Array(4 * n ** 2);
    let wave = new Float32Array(n ** 2);
    let gl = this.gl;
    gl.readPixels(0, 0, n, n, gl.RGBA, gl.FLOAT, rgba);
    for (let i = 0; i < n ** 2; i++)
      wave[i] = rgba[i * 4];
    return wave;
  }
}

/**
 * Computes b in the Au=b implicit scheme:
 * 
 *    u_tt = c^2 * Du + g * u_t
 * 
 * Approximation is done for u(t=k*dt) over t=(k-1)*dt..(k+1)*dt
 * using 2nd order FTD schemes:
 * 
 *    u_tt = 1/dt^2 * (u(k+1) - 2*u(k) + u(k-1))
 *    u_t = 1/2dt * (u(k+1) - u(k-1))
 *    Du = 1/6 * (Du(k+1) + 6*Du(k) + Du(k-1))
 *
 * Where Du is the 9 point laplacian stencil:
 * 
 *    0.25 0.50 0.25
 *    0.50  -3  0.50
 *    0.25 0.50 0.25
 * 
 * In the Au=b, A is the matrix coefficients at u(k+1) and b is
 * everything else (the averaged u(k) and u(k-1) in practice).
 */
class GpuWaveImplicitRhsProgram {
  constructor(gpuContext, { speed, dx, dt }) {
    let gl = gpuContext.gl;

    this.gl = gl;
    this.dx = dx;

    this.vertexShader = GpuLaplaceVertexShader.get(gl);

    this.fragmentShader = GpuProgram.createFragmentShader(gl, `
        precision highp float;
        precision highp sampler2D;

        varying vec2 vUv;

        varying vec2 vL;
        varying vec2 vR;
        varying vec2 vT;
        varying vec2 vB;

        varying vec2 vLT;
        varying vec2 vRT;
        varying vec2 vLB;
        varying vec2 vRB;        

        uniform sampler2D uPrev1;
        uniform sampler2D uPrev2;
        uniform vec2 g;

        ${GpuWaveProgram.utils}

        void main () {
          float a2 = ${(speed * dt / dx) ** 2}; // = 1/4
          vec2 gt = ${0.5 * dt}*g; // |gt| < 2/dt

          vec2 u1 = texture2D(uPrev1, vUv).xy;
          vec2 u2 = texture2D(uPrev2, vUv).xy;

          vec2 d1 = diffuse(uPrev1);
          vec2 d2 = diffuse(uPrev2);

          vec2 b =
            + u1 + (u1 - u2)
            - xmul(gt, u2)
            + 2.0/3.0 * a2 * d1
            + 1.0/6.0 * a2 * d2;

          gl_FragColor = vec4(b, 0.0, 0.0);
        }
    `);

    this.gp = new GpuProgram(
      this.gl,
      this.vertexShader,
      this.fragmentShader);
  }

  run({ force, wave1, wave2, output }) {
    let vars = this.gp.uniforms;
    this.gp.bind();
    this.gl.uniform2f(vars.g, force[0], force[1]);
    this.gl.uniform1f(vars.dx, this.dx);
    this.gl.uniform1i(vars.uPrev1, wave1.attach(0));
    this.gl.uniform1i(vars.uPrev2, wave2.attach(1));
    GpuProgram.blit(this.gl, output);
  }
}

// Does one iteration u = 1/y*(b + Bu)
// where A = y - B, and Au = b.
class GpuWaveImplicitStepProgram {
  constructor(gpuContext, { speed, dx, dt }) {
    let gl = gpuContext.gl;

    this.gl = gl;
    this.dx = dx;

    this.vertexShader = GpuLaplaceVertexShader.get(gl);

    this.fragmentShader = GpuProgram.createFragmentShader(gl, `
        precision highp float;
        precision highp sampler2D;

        varying vec2 vUv;

        varying vec2 vL;
        varying vec2 vR;
        varying vec2 vT;
        varying vec2 vB;

        varying vec2 vLT;
        varying vec2 vRT;
        varying vec2 vLB;
        varying vec2 vRB;        

        uniform sampler2D uRhs; // b in Ax=b
        uniform sampler2D uWave; // x in Ax=b
        uniform vec2 g;

        ${GpuWaveProgram.utils}

        void main () {
          float a2 = ${(speed * dt / dx) ** 2}; // = 1/4
          vec2 gt = ${0.5 * dt}*g; // |gt| < 2/dt

          vec2 b1 = texture2D(uRhs, vUv).xy;
          vec2 u1 = texture2D(uWave, vUv).xy;
          vec2 d1 = diffuse(uWave);
          vec2 y1 = vec2(1.0 + 0.5*a2, 0.0) - gt;
          vec2 u2 = b1 + a2/6.0 * (d1 + 3.0*u1);

          gl_FragColor = vec4(xdiv(u2, y1), 0.0, 0.0);
        }
    `);

    this.gp = new GpuProgram(
      this.gl,
      this.vertexShader,
      this.fragmentShader);
  }

  run({ force, input, rhs, output }) {
    let vars = this.gp.uniforms;
    this.gp.bind();
    this.gl.uniform2f(vars.g, force[0], force[1]);
    this.gl.uniform1f(vars.dx, this.dx);
    this.gl.uniform1i(vars.uRhs, rhs.attach(0));
    this.gl.uniform1i(vars.uWave, input.attach(1));
    GpuProgram.blit(this.gl, output);
  }
}

// Runs the implicit scheme solver with N iterations.
class GpuWaveImplicitProgram {
  get output() {
    return this.frames[0];
  }

  constructor(gpuContext, { boundary, size, speed, dx, dt, iterations }) {
    console.log(`Implicit wave solver: ${iterations} iterations`);
    let gl = gpuContext.gl;

    // It's weird, but just 3 iterations seems enough.
    // It's more common to use 20-30 iterations, though.
    if (iterations < 3)
      throw new Error('Need at least 3 iterations');

    this.gl = gl;
    this.size = size;
    this.dx = dx;
    this.boundary = boundary;
    this.iterations = iterations;

    this.gpRhs = new GpuWaveImplicitRhsProgram(
      gpuContext, { speed, dx, dt });

    this.gpStep = new GpuWaveImplicitStepProgram(
      gpuContext, { speed, dx, dt });

    this.rhs = gpuContext.createFrameBuffer(size, 2);
    this.wave = gpuContext.createFrameBuffer(size, 2);

    this.frames = [
      gpuContext.createFrameBuffer(size, 2),
      gpuContext.createFrameBuffer(size, 2),
      gpuContext.createFrameBuffer(size, 2),
    ];

    this.vertexShader = GpuSingleVertexShader.get(gl);

    this.fragmentShader = GpuProgram.createFragmentShader(gl, `
        precision highp float;
        precision highp sampler2D;

        varying vec2 vUv;

        uniform sampler2D uBoundary;
        uniform sampler2D uWave;

        void main () {
            float b1 = texture2D(uBoundary, vUv).r;
            vec2 u1 = texture2D(uWave, vUv).xy;
            
            gl_FragColor = vec4(u1 * (1.0 - b1), 0.0, 0.0);
        }
    `);

    this.gp = new GpuProgram(
      this.gl,
      this.vertexShader,
      this.fragmentShader);
  }

  applyBC() {
    let gl = this.gl;
    let gp = this.gp;
    let vars = gp.uniforms;
    gp.bind();
    gl.uniform1i(vars.uBoundary, this.boundary.attach(0));
    gl.uniform1i(vars.uWave, this.wave.attach(1));
    GpuProgram.blit(gl, this.frames[0]);
  }

  run(force) {
    let [f0, f1, f2] = this.frames;
    this.frames = [f2, f0, f1];

    // Compute b in Ax=b.
    this.gpRhs.run({
      force,
      wave1: f0,
      wave2: f1,
      output: this.rhs,
    });

    // Use iterative solver for Ax=b.
    let w1 = this.wave;
    let w2 = this.frames[0];

    for (let i = 0; i < this.iterations; i++) {
      this.gpStep.run({
        force,
        rhs: this.rhs,
        input: i == 0 ? this.frames[1] : w1,
        output: w2,
      });

      [w1, w2] = [w2, w1];
    }

    this.wave = w1;
    this.frames[0] = w2;

    // Apply the boundary condition.
    this.applyBC();
  }

  read() {
    let n = this.size;
    let rgba = new Float32Array(4 * n ** 2);
    let wave = new Float32Array(n ** 2);
    let gl = this.gl;
    gl.readPixels(0, 0, n, n, gl.RGBA, gl.FLOAT, rgba);
    for (let i = 0; i < n ** 2; i++)
      wave[i] = rgba[i * 4];
    return wave;
  }
}

class GpuDisplayProgram {
  constructor(gl, { stats, size }) {
    this.gl = gl;

    this.stats = stats;

    this.vertexShader = GpuProgram.createVertexShader(gl, `
      precision highp float;
      attribute vec2 aPosition;
      varying vec2 vUv;

      void main () {
          // The texture2D() coordinates are in the 0..1 range,
          // hence mapping from -1..+1 to 0..1. All math operations
          // map this vUv back to the -1 < x < +1 space.
          vUv = aPosition * 0.5 + 0.5;

          // The wave texture is mapped 1:1 to the canvas.
          gl_Position = vec4(aPosition, 0.0, 1.0);
      }
    `);

    this.fragmentShader = GpuProgram.createFragmentShader(gl, `
        precision highp float;
        precision highp sampler2D;

        varying vec2 vUv;

        uniform sampler2D uAmpStats;
        uniform sampler2D uWave;

        // All components are in the range 0..1, including hue.
        vec3 hsv2rgb(vec3 c) {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }        

        void main () {
            vec4 stats = texture2D(uAmpStats, vec2(0.0, 0.0));
            float rdev = sqrt(stats.w) / ${size.toFixed(1)};
            vec2 u = texture2D(uWave, vUv).xy;

            float rad = length(u) / (rdev * 3.0);
            float arg = atan(u.y, u.x) / ${2 * Math.PI} + 0.5;

            vec3 rgb = hsv2rgb(vec3(arg, 1.0, rad));
            gl_FragColor = vec4(rgb, 1.0);
        }
    `);

    this.gp = new GpuProgram(
      this.gl,
      this.vertexShader,
      this.fragmentShader);
  }

  run(input) {
    let gl = this.gl;
    let gp = this.gp;
    let uf = gp.uniforms;
    gp.bind();
    gl.uniform1i(uf.uWave, input.attach(0));
    gl.uniform1i(uf.uAmpStats, this.stats.attach(1));
    GpuProgram.blit(gl, null); // null = canvas
  }
}

class GpuSplashProgram {
  constructor(gl) {
    this.gl = gl;

    this.vertexShader = GpuSingleVertexShader.get(gl);

    this.fragmentShader = GpuProgram.createFragmentShader(gl, `
        precision highp float;

        varying vec2 vUv;
        uniform vec2 point;
        uniform float radius;
        uniform float thickness;
        uniform float amplitude;
        uniform float angularFreq;
        uniform float radialFreq;

        void main () {
            vec2 v = vUv * 2.0 - 1.0 - point;
            float r = length(v);
            float d = r - radius;
            float h = exp(-1.0/thickness * d*d);
            float a = r > 0.0 ? acos(v.x / r) * sign(v.y) : 0.0;
            float z = amplitude * h 
              * cos(a * angularFreq)
              * cos(d * radialFreq * ${2 * Math.PI});
            gl_FragColor = vec4(z, 0.0, 0.0, 0.0);
        }
    `);

    this.gp = new GpuProgram(
      this.gl,
      this.vertexShader,
      this.fragmentShader);
  }

  run({ output,
    amplitude = 0.01,
    thickness = 1e-3,
    radius = 1.0,
    angularFreq = 0,
    radialFreq = 0,
    x = 0, y = 0 }) {

    let gl = this.gl;
    let gp = this.gp;
    let uf = gp.uniforms;

    gp.bind();
    gl.uniform2f(uf.point, x, -y);
    gl.uniform1f(uf.radius, radius);
    gl.uniform1f(uf.thickness, thickness);
    gl.uniform1f(uf.amplitude, amplitude);
    gl.uniform1f(uf.angularFreq, angularFreq);
    gl.uniform1f(uf.radialFreq, radialFreq);
    GpuProgram.blit(gl, output);
  }
}

class GpuBoundaryProgram {
  constructor(gpuContext, size) {
    let gl = gpuContext.gl;

    this.gl = gl;
    this.size = size;
    this.output = gpuContext.createFrameBuffer(size);

    this.vertexShader = GpuSingleVertexShader.get(gl);

    this.fragmentShader = GpuProgram.createFragmentShader(gl, `
        precision highp float;

        varying vec2 vUv;
        uniform vec2 center;
        uniform float radius;
        uniform float sharpness;
        uniform float thickness;

        void main () {
            vec2 v = vUv * 2.0 - 1.0 - center;
            float r = length(v);
            float d = r - radius;
            float e = d > -sharpness ? 1.0 :
              d < -2.0*sharpness ? 0.0 :
              exp(-1.0/thickness * d*d);
            gl_FragColor = vec4(e);
        }
    `);

    this.gp = new GpuProgram(
      this.gl,
      this.vertexShader,
      this.fragmentShader);
  }

  run({ radius, sharpness, thickness }) {
    let gl = this.gl;
    let gp = this.gp;
    let uf = gp.uniforms;
    gp.bind();
    gl.uniform2f(uf.center, 0, 0);
    gl.uniform1f(uf.radius, radius);
    gl.uniform1f(uf.sharpness, sharpness);
    gl.uniform1f(uf.thickness, thickness);
    GpuProgram.blit(gl, this.output);
  }
}

// output[i] = (input[i] + shift) * scale
class GpuRescalingProgram {
  constructor(gpuc) {
    this.gpuc = gpuc;

    this.vertexShader = GpuSingleVertexShader.get(gpuc.gl);

    this.fragmentShader = gpuc.createFragmentShader(`
        precision highp float;

        varying vec2 vUv;

        uniform sampler2D uData;
        uniform vec4 shift;
        uniform float scale;

        void main () {
            vec4 u = texture2D(uData, vUv);
            gl_FragColor = (u + shift) * scale;
        }
    `);

    this.gp = gpuc.createProgram(
      this.vertexShader,
      this.fragmentShader);
  }

  run({ output, input, shift = [0, 0], scale = 1.0 }) {
    let gl = this.gpuc.gl;
    let vars = this.gp.uniforms;
    this.gp.bind();
    gl.uniform4f(vars.shift, shift[0], shift[1], 0, 0);
    gl.uniform1f(vars.scale, scale);
    gl.uniform1i(vars.uData, input.attach(0));
    GpuProgram.blit(gl, output);
  }
}

class GpuStatsProgram {
  constructor(gpuContext, { size, dx }) {
    this.gl = gpuContext.gl;
    this.size = size;
    this.dx = dx;

    this.buffers = [];
    for (let i = 0; 2 ** i <= size; i++)
      this.buffers[i] = gpuContext.createFrameBuffer(2 ** i, 4);
    console.log('Created', this.buffers.length,
      'stats buffers for', size, 'x', size, 'inputs');

    this.output = this.buffers[0]; // 1x1

    let gl = this.gl;

    this.vertexShader = GpuProgram.createVertexShader(gl, `
      precision highp float;
      attribute vec2 aPosition;

      varying vec2 v1;
      varying vec2 v2;
      varying vec2 v3;
      varying vec2 v4;

      uniform float dx;

      void main () {
          float dx2 = dx * 0.5;
          vec2 v = aPosition * 0.5 + 0.5;

          v1 = v + vec2(-dx2, -dx2);
          v2 = v + vec2(-dx2, +dx2);
          v3 = v + vec2(+dx2, +dx2);
          v4 = v + vec2(+dx2, -dx2);

          gl_Position = vec4(aPosition, 0.0, 1.0);
      }
    `);

    // s.xy = avg(u)
    // s.z = max(length(u))
    // s.w = variance(length(u))
    this.fragmentShader = GpuProgram.createFragmentShader(gl, `
      precision highp float;
      precision highp sampler2D;

      varying vec2 v1;
      varying vec2 v2;
      varying vec2 v3;
      varying vec2 v4;

      uniform sampler2D uData;
      // output = count*4, at the top layer,
      // count=0 and dx=0, so v1..v4 are same.
      uniform float count;

      // en.wikipedia.org/wiki/Algorithms_for_calculating_variance
      vec4 merge(vec4 u1, vec4 u2, float n) {
        vec2 d = u1.xy - u2.xy;
        vec2 avg = 0.5*(u1.xy + u2.xy);
        float var = u1.w + u2.w + dot(d, d)*n*0.5;
        
        return vec4(
          avg,
          max(u1.z, u2.z),
          var);
      }

      void main () {
        if (count < 1.0) {
          // avg = u, max = r, var = 0
          vec2 u = texture2D(uData, v1).xy;
          float r = length(u);
          gl_FragColor = vec4(u, r, 0.0);
          return;
        }

        vec4 u1 = texture2D(uData, v1);
        vec4 u2 = texture2D(uData, v2);
        vec4 u3 = texture2D(uData, v3);
        vec4 u4 = texture2D(uData, v4);

        vec4 u12 = merge(u1, u2, count);
        vec4 u34 = merge(u3, u4, count);

        gl_FragColor = merge(u12, u34, count*2.0);
      }
    `);

    this.gp = new GpuProgram(
      this.gl,
      this.vertexShader,
      this.fragmentShader);
  }

  fold({ dx, count, input, output }) {
    let gl = this.gl;
    let gp = this.gp;
    let uf = gp.uniforms;
    gp.bind();
    gl.uniform1f(uf.dx, dx);
    gl.uniform1f(uf.count, count);
    gl.uniform1i(uf.uData, input.attach(0));
    GpuProgram.blit(gl, output);
  }

  run(input) {
    let depth = this.buffers.length;

    this.fold({
      dx: 0,
      count: 0,
      input,
      output: this.buffers[depth - 1],
    });

    for (let i = 0; i < depth - 1; i++) {
      this.fold({
        dx: this.dx * 2 ** i,
        count: 4 ** i,
        input: this.buffers[depth - i - 1],
        output: this.buffers[depth - i - 2],
      });
    }
  }

  read() {
    let res = new Float32Array(4);
    let gl = this.gl;
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, res);
    return [...res];
  }
}

class GpuSWaveProgram {
  get output() {
    return this.frames[0];
  }

  constructor(gpuContext, { boundary, size, dx, dt }) {
    console.log('Explicit S wave solver:', size, 'dt', dt);
    let gl = gpuContext.gl;

    this.gl = gl;
    this.size = size;
    this.dx = dx;
    this.boundary = boundary;

    this.frames = [
      gpuContext.createFrameBuffer(size, 2),
      gpuContext.createFrameBuffer(size, 2)];

    this.vertexShader = GpuLaplaceVertexShader.get(gl);

    this.fragmentShader = GpuProgram.createFragmentShader(gl, `
        precision highp float;
        precision highp sampler2D;

        varying vec2 vUv;

        varying vec2 vL;
        varying vec2 vR;
        varying vec2 vT;
        varying vec2 vB;

        varying vec2 vLT;
        varying vec2 vRT;
        varying vec2 vLB;
        varying vec2 vRB;        

        uniform sampler2D uBoundary;
        uniform sampler2D uPrev;
        uniform vec2 gPrev;

        ${GpuWaveProgram.utils}

        void main () {
            vec2 v = vUv * 2.0 - 1.0;
            float p = 1.0 / (0.1 + dot(v, v));
            float c = ${(dt / dx ** 2).toExponential()};
            float b = texture2D(uBoundary, vUv).r;
            vec2 u1 = texture2D(uPrev, vUv).xy;
            vec2 diff = diffuse(uPrev);
            vec2 du = 5e-4*c*diff - ${dt}*xmul(gPrev, u1)*p;
            vec2 u0 = u1 + vec2(-du.y, du.x); // du*i, i*i = -1
            
            gl_FragColor = vec4(u0*(1.0 - b), 0.0, 0.0);
        }
    `);

    this.gp = new GpuProgram(
      this.gl,
      this.vertexShader,
      this.fragmentShader);
  }

  run(force) {
    let [f0, f1] = this.frames;
    this.frames = [f1, f0];

    let gl = this.gl;
    let gp = this.gp;
    let uf = gp.uniforms;
    gp.bind();
    gl.uniform2f(uf.gPrev, force[0], 0*force[1]);
    gl.uniform1f(uf.dx, this.dx);
    gl.uniform1i(uf.uBoundary, this.boundary.attach(0));
    gl.uniform1i(uf.uPrev, f0.attach(1));
    GpuProgram.blit(gl, f1);
  }
}