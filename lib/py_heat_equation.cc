#include "heat_equation.cc"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

// -------- PYTHON BINDINGS
// -----------------------------------------------------------------------

namespace py = pybind11;
using namespace Heat_Transfer;

PYBIND11_PLUGIN(dealii_heat_equation)
{
  py::module m("dealii_heat_equation", "deal.II heat example");

  py::class_<HeatEquation<2>>(m, "HeatExample")
    .def("run", &HeatEquation<2>::run);

  return m.ptr();
}