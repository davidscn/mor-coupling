# CMake minimum version
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

# Set the name of the project and target:
SET(TARGET "heat_equation")
   SET(_src
   ${TARGET}.cc
)

# Set pybind version
ADD_SUBDIRECTORY(pybind11)

DEAL_II_INITIALIZE_CACHED_VARIABLES()

pybind11_add_module(dealii_${TARGET} py_${TARGET}.cc ${TARGET}.cc)

DEAL_II_SETUP_TARGET(dealii_${TARGET})

FIND_PACKAGE(precice 2.0 REQUIRED)
TARGET_LINK_LIBRARIES(dealii_${TARGET} precice::precice)