#!/bin/bash

set -eux
set -o pipefail
PRECICE_VERSION=2.3.0
sudo apt-get -qy update
wget -q https://github.com/precice/precice/releases/download/v${PRECICE_VERSION}/libprecice2_${PRECICE_VERSION}_focal.deb
sudo apt-get -qy install ./libprecice2_${PRECICE_VERSION}_focal.deb python3-pip python-is-python3
sudo apt-get -qy install fenics
python3 -m pip install pymor~=2021.1
cd /src/pymor-deal.II
python3 -m pip install .
python3 -m pip install fenicsprecice
cd /src/mor-coupling
cmake .
make
/src/mor-coupling/example/dirichlet-fenics/run.sh &
# implicitly add dir to python path
cd /src/mor-coupling/example/neumann-reduced/
python3 heat_equation_reduced.py
