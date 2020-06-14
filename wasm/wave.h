#include <vector>

// GPU uses 32-bit floats.
typedef float xfloat_t;

// u_tt = c^2*(u_xx + y_yy) + g(t)*u_t
// g(t) = -10 + a*cos(b*t)
// u = 0 at x^2+y^2=r^2, the boundary condition
class WaveSolver {
public:
  // 0 < x < grid_size * grid_step
  // 0 < y < grid_size * grid_step
  WaveSolver(
    int grid_size,
    xfloat_t grid_step,
    xfloat_t time_step,
    xfloat_t wave_speed);

  void init();
  void use_init_wave();

  // Advances the wave by wave_speed*time_step
  // and sets the boundary condition.
  void next(xfloat_t input_amplitude);

  // Row by row, bottom up:
  //
  //  y2  6  7  8
  //  y1  3  4  5
  //  y0  0  1  2
  //      x0 x1 x2
  //
  // In other words, (xi, yj) maps to
  // grid[i+j*N], where N=grid_size.
  size_t wave() {
    return (size_t)grid_plane(0).data();
  }

  size_t edge() {
    return (size_t)m_edge.data();
  }

  // Same as wave(), but converted to a RGBAxNxN buffer.
  // Can be applied directly with putImageData().
  size_t rgba();
  size_t isoline();

private:
  int m_grid_size;
  xfloat_t m_grid_step;
  xfloat_t m_time_step;
  xfloat_t m_wave_speed;
  int m_steps; // time = m_steps * m_time_step
  std::vector<std::vector<xfloat_t>> m_planes;
  std::vector<xfloat_t> m_edge;
  std::vector<uint8_t> m_rgba;

  xfloat_t m_pmin;
  xfloat_t m_pmax;
  xfloat_t m_pavg;
  xfloat_t m_pdev;
  xfloat_t m_vmin;
  xfloat_t m_vmax;
  xfloat_t m_vavg;
  xfloat_t m_vdev;

  std::vector<xfloat_t> & grid_plane(int time_index);
  int grid_index(int i, int j);
  void update_stats();
  void update_edge();
};
