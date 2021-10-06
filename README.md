# mor-coupling
Coupling MOR codes using pyMOR and preCICE

## Installation

Install [`preCICE`](https://precice.org/installation-overview.html), [`pyMOR`](https://github.com/pymor/pymor#installation-via-pip) and [`deal.II`](https://dealii.org/current/readme.html) (v9.2 or greater). Afterwards, install the [pyMOR-deal.II wrapper](https://github.com/pymor/pymor-deal.II) using

```
git clone git@github.com:pymor/pymor-deal.II.git
cd pymor-deal.II
git submodule init
git submodule update
pip install .
```
and clone this repository using

```
git clone git@github.com:DavidSCN/mor-coupling.git
cd mor-coupling
git submodule init
git submodule update
```
