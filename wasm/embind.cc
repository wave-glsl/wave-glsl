#include <emscripten/bind.h>
#include "wave.h"

using namespace emscripten;

EMSCRIPTEN_BINDINGS(wave_solver) {
  class_<WaveSolver>("WaveSolver")
    .constructor<int, double, double, double>()
    .function("init", &WaveSolver::init)
    .function("next", &WaveSolver::next)
    .function("wave", &WaveSolver::wave)
    .function("edge", &WaveSolver::edge)
    .function("rgba", &WaveSolver::rgba)
    .function("isoline", &WaveSolver::isoline)
    .function("use_init_wave", &WaveSolver::use_init_wave)
    ;
}
