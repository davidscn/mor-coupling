#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include "heat_equation.cc"

// TODO:
#include <adapter/parameters.cc>

// -------- PYTHON BINDINGS
// -----------------------------------------------------------------------

namespace py = pybind11;
using namespace Heat_Transfer;

PYBIND11_MODULE(dealii_heat_equation, m)
{
  m.doc() = "deal.II heat example";

  py::class_<HeatEquation<2>>(m, "HeatExample")
    .def(py::init<const std::string &>(),
         py::arg("parameter_file") = std::string("parameters"))
    .def("make_grid", &HeatEquation<2>::make_grid)
    .def("setup_system", &HeatEquation<2>::setup_system)
    .def("advance", &HeatEquation<2>::advance)
    .def("stationary_matrix",
         &HeatEquation<2>::stationary_system_matrix,
         py::return_value_policy::reference_internal)
    .def("set_initial_condition", &HeatEquation<2>::set_initial_condition)
    .def("initialize_precice", &HeatEquation<2>::initialize_precice)
    .def("output_results", &HeatEquation<2>::output_results)
    .def("assemble_rhs", &HeatEquation<2>::assemble_rhs)
    .def("is_coupling_ongoing", &HeatEquation<2>::is_coupling_ongoing);
}
