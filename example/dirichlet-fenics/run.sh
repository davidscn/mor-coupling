#!/bin/bash
set -e -u

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" && cd . && pwd -P )"
cd "${THIS_DIR}"

python3 heat.py -d
