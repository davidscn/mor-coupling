#!/bin/bash

set -eux
set -o pipefail


apt update -q
apt install -yq libgsl-dev libarpack2-dev liblapack-dev libmuparser-dev libmetis-dev libtbb-dev

python3 -m pip install fenicsprecice pymor~=2021.2

# the pyMOR docker ecosystem compiles extension modules with
# the oldest supported numpy version, but the final
# testing image can have a newer version installed
# therefore we must install the potentially older version here again
# otherwise we'll get segfaults when extension modules for
# fenics, deal.II and our coupling with differing numpy 
# versions are loaded
python3 -m pip install pymor-oldest-supported-numpy~=2021.1.0

cd /src/mor-coupling
cmake .
make
/src/mor-coupling/example/dirichlet-fenics/run.sh &
# implicitly add dir to python path
cd /src/mor-coupling/example/neumann-reduced/
python3 heat_equation_reduced.py
