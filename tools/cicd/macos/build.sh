#!/usr/bin/env bash

brew install gcc
export CC=/usr/local/Cellar/gcc/9.2.0_3/bin/gcc-9
export CXX=/usr/local/Cellar/gcc/9.2.0_3/bin/g++-9
export CPP=/usr/local/Cellar/gcc/9.2.0_3/bin/cpp-9

export GBKFIT_BUILD_OPENMP=1

. venv/bin/activate

python -m pip install cmake

python -m pip wheel . -w wheels --no-deps
