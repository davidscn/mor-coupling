# mor-coupling
Coupling MOR codes using pyMOR and preCICE

## Installation

Install [`preCICE`](https://precice.org/installation-overview.html), [`pyMOR`](https://github.com/pymor/pymor#installation-via-pip) and [`deal.II`](https://dealii.org/current/readme.html) (v9.2 or greater). Afterwards, install the [pyMOR-deal.II wrapper](https://github.com/pymor/pymor-deal.II) using

```
git clone --recurse-submodules https://github.com/pymor/pymor-deal.II.git
python3 -m pip install ./pymor-deal.II
```
and clone this repository using

```
git clone --recurse-submodules https://github.com/DavidSCN/mor-coupling.git
cmake -B ./mor-couling/build -S ./mor-coupling
cmake --build ./mor-couling/build 
```
