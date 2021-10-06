##
#  CMake script for the dealii-adapter:
##

# CMake minimum version
CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

# Set pybind version
SET(PYBIND11_PYTHON_VERSION "3" CACHE STRING "Python version to use for compiling the example library")

ADD_SUBDIRECTORY(pybind11)
# Set the name of the project and target:
SET(TARGET "heat_equation")
SET(_src
    ${TARGET}.cc
)

pybind11_add_module(dealii_${TARGET} py_${TARGET}.cc ${TARGET}.cc)

DEAL_II_SETUP_TARGET(dealii_${TARGET})

FIND_PACKAGE(precice REQUIRED)
TARGET_LINK_LIBRARIES(dealii_${TARGET} precice::precice)

