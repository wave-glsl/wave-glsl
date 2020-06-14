#include <stdio.h>
#include "wave.h"

#define GID(i, j, n)  (((i) + (n)) % (n) + ((j) + (n)) % (n) * (n))

xfloat_t sqr(xfloat_t x) {
  return x * x;
}

xfloat_t sigmoid(xfloat_t x) {
  return 2.0 / (1.0 + exp(-x)) - 1.0;
}

xfloat_t rand_minmax(xfloat_t min, xfloat_t max) {
  return xfloat_t(rand()) / xfloat_t(RAND_MAX) * (max - min) + min;
}

WaveSolver::WaveSolver(
    int grid_size,
    xfloat_t grid_step,
    xfloat_t time_step,
    xfloat_t wave_speed) {

  printf("N=%d dx=%.2e dt=%.2e c=%.2e\n",
    grid_size, grid_step, time_step, wave_speed);

  printf("c*dt/dx=%.3f\n",
    wave_speed*time_step/grid_step);

  assert(grid_size > 0);
  assert(grid_step > 0);
  assert(time_step > 0);
  assert(wave_speed > 0);
  assert(wave_speed*time_step/grid_step <= 0.5);

  m_grid_size = grid_size;
  m_grid_step = grid_step;
  m_time_step = time_step;
  m_wave_speed = wave_speed;
}

void WaveSolver::init() {
  size_t n2 = m_grid_size * m_grid_size;

  m_steps = 0;
  
  m_edge.resize(n2);
  m_rgba.resize(n2 * 4);
  m_planes.resize(3);

  for (auto & plane : m_planes)
    plane.resize(n2);
}

void WaveSolver::use_init_wave() {
  auto & wave = grid_plane(0);

  for (auto & grid : m_planes)
    for (size_t i = 0; i < grid.size(); i++)
      grid[i] = wave[i];
}

void WaveSolver::update_edge() {
  size_t n = m_grid_size;
  size_t n2 = n * n;

  xfloat_t* p0 = grid_plane(0).data();
  xfloat_t* ed = m_edge.data();

  for (size_t i = 0; i < n2; i++) {
    xfloat_t x = ed[i];
    if (x > 0)
      p0[i] *= (1 - x);
  }
}

void WaveSolver::next(xfloat_t input_amplitude) {
  m_steps++;

  xfloat_t* p0 = grid_plane(0).data();
  xfloat_t* p1 = grid_plane(1).data();
  xfloat_t* p2 = grid_plane(2).data();

  int n = m_grid_size;

  xfloat_t dt = m_time_step;
  xfloat_t dh = m_grid_step;

  xfloat_t g2 = 0.5 * dt * input_amplitude;
  xfloat_t c2 = sqr(m_wave_speed * dt / dh);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      size_t ij = GID(i, j, n);

      // See the 9-point stencil for the discretized Laplacian:
      // en.wikipedia.org/wiki/Discrete_Laplace_operator
      xfloat_t diff =
        + 0.50 * p1[GID(i + 1, j,     n)]
        + 0.50 * p1[GID(i - 1, j,     n)]
        + 0.50 * p1[GID(i,     j + 1, n)]
        + 0.50 * p1[GID(i,     j - 1, n)]
        + 0.25 * p1[GID(i - 1, j - 1, n)]
        + 0.25 * p1[GID(i - 1, j + 1, n)]
        + 0.25 * p1[GID(i + 1, j + 1, n)]
        + 0.25 * p1[GID(i + 1, j - 1, n)]
        -3 * p1[ij];

      xfloat_t u2 = 2 * p1[ij]
        - (1 + g2) * p2[ij]
        + c2 * diff;

      p0[ij] = u2 / (1 - g2);
    }
  }

  update_edge();
}

void WaveSolver::update_stats() {
  size_t n2 = m_grid_size * m_grid_size;

  xfloat_t* wave = grid_plane(0).data();
  xfloat_t* prev = grid_plane(1).data();

  m_pavg = wave[0]/n2;
  m_pmin = m_pavg;
  m_pmax = m_pavg;

  m_vavg = (wave[0] - prev[0]) / m_time_step / n2;
  m_vmin = m_vavg;
  m_vmax = m_vavg;

  for (size_t i = 0; i < n2; i++) {
    xfloat_t p = wave[i];
    xfloat_t v = (wave[i] - prev[i]) / m_time_step;
    m_pavg += p / n2;
    m_vavg += v / n2;
    m_pmin = fmin(m_pmin, p);
    m_pmax = fmax(m_pmax, p);
    m_vmin = fmin(m_vmin, v);
    m_vmax = fmax(m_vmax, v);
  }

  m_pdev = 0;
  m_vdev = 0;

  for (size_t i = 0; i < n2; i++) {
    xfloat_t p = wave[i];
    xfloat_t v = (wave[i] - prev[i]) / m_time_step;
    m_pdev += sqr(p - m_pavg)/n2;
    m_vdev += sqr(v - m_vavg)/n2;
  }

  m_pdev = sqrt(m_pdev);
  m_vdev = sqrt(m_vdev);

  if (!m_pdev) m_pdev = 1;
  if (!m_vdev) m_vdev = 1;
}

size_t WaveSolver::rgba() {
  size_t n2 = m_grid_size * m_grid_size;

  xfloat_t* wave = grid_plane(0).data();
  xfloat_t* prev = grid_plane(1).data();

  update_stats();

  for (int i = 0; i < n2; i++) {
    xfloat_t p = wave[i];
    xfloat_t v = (wave[i] - prev[i]) / m_time_step;

    xfloat_t amp = (p - m_pavg) / (3 * m_pdev);
    xfloat_t vel = (v - m_vavg) / (3 * m_vdev);

    m_rgba[i*4 + 0] = (uint8_t)(fmax(0, fmin(1, +amp)) * 255);
    m_rgba[i*4 + 1] = (uint8_t)(fmax(0, fmin(1, -amp)) * 255);
    m_rgba[i*4 + 2] = (uint8_t)(fmax(0, fmin(1, fabs(vel))) * 255);;
    m_rgba[i*4 + 3] = 255; // opacity
  }

  return (size_t)m_rgba.data();
}

size_t WaveSolver::isoline() {
  size_t n = m_grid_size;
  xfloat_t* wave = grid_plane(0).data();
  update_stats();

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      xfloat_t l = wave[GID(i - 1, j, n)] - m_pavg;
      xfloat_t r = wave[GID(i + 1, j, n)] - m_pavg;
      xfloat_t t = wave[GID(i, j - 1, n)] - m_pavg;
      xfloat_t b = wave[GID(i, j + 1, n)] - m_pavg;

      xfloat_t max = fmax(fmax(l, r), fmax(t, b));
      xfloat_t min = fmin(fmin(l, r), fmin(t, b));

      uint8_t c = min < 0 && max > 0 ? 0 : 255;

      size_t p = GID(i, j, n);

      m_rgba[p*4 + 0] = c;
      m_rgba[p*4 + 1] = c;
      m_rgba[p*4 + 2] = c;
      m_rgba[p*4 + 3] = 255; // opacity
    }
  }

  return (size_t)m_rgba.data();
}

std::vector<xfloat_t> & WaveSolver::grid_plane(int time_index) {
  return m_planes[(3 + m_steps - time_index) % 3];
}

int WaveSolver::grid_index(int i, int j) {
  return i + j * m_grid_size;
}
