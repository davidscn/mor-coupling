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
    .def("make_grid_and_sparsity_pattern",
         &HeatEquation<2>::make_grid_and_sparsity_pattern)
    .def("create_system_matrix",
         &HeatEquation<2>::create_system_matrix,
         py::arg("coefficient1"),
         py::arg("coefficient2"),
         py::arg("threshold_x"),
         py::arg("threshold_y"),
         py::return_value_policy::move)
    .def("advance", &HeatEquation<2>::advance)
    .def("set_initial_condition", &HeatEquation<2>::set_initial_condition)
    .def("initialize_precice", &HeatEquation<2>::initialize_precice)
    .def("output_results", &HeatEquation<2>::output_results)
    .def("assemble_rhs", &HeatEquation<2>::assemble_rhs)
    .def("is_coupling_ongoing", &HeatEquation<2>::is_coupling_ongoing);
}
