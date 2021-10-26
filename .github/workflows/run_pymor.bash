#!/bin/bash

set -eux
set -o pipefail


apt update -q
apt install -yq libgsl-dev libarpack2-dev liblapack-dev libmuparser-dev libmetis-dev libtbb-dev
python3 -m pip install fenicsprecice pymor~=2021.1
cd /src/mor-coupling
cmake .
make
/src/mor-coupling/example/dirichlet-fenics/run.sh &
# implicitly add dir to python path
cd /src/mor-coupling/example/neumann-reduced/
python3 heat_equation_reduced.py
