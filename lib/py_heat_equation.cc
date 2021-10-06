// This file is part of the pyMOR project (http://www.pymor.org).
// Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include "heat_equation.cc"

// -------- PYTHON BINDINGS
// -----------------------------------------------------------------------

namespace py = pybind11;
using namespace Heat_Transfer;

PYBIND11_PLUGIN(dealii_heat_equation)
{
  py::module m("dealii_heat_equation", "deal.II heat example");

  py::class_<HeatEquation<2>>(m, "HeatExample")
    .def("stationary_system_matrix", &HeatEquation<2>::stationary_system_matrix, py::return_value_policy::automatic_reference)
    .def("run", &HeatEquation<2>::run);

  return m.ptr();
}