#!/usr/bin/env bash

brew install fftw libomp

#export CC=/usr/local/Cellar/gcc/9.2.0_3/bin/gcc-9
#export CXX=/usr/local/Cellar/gcc/9.2.0_3/bin/g++-9
#export CPP=/usr/local/Cellar/gcc/9.2.0_3/bin/cpp-9

export MACOSX_DEPLOYMENT_TARGET=10.9
export GBKFIT_BUILD_HOST=1

. venv/bin/activate

python -m pip install cmake

python -m pip wheel . -w wheels --no-deps

python -m pip install delocate

ls wheels

for whl in wheels/*.whl; do
    delocate-wheel -w wheels_fixed -v "$whl"
done

ls wheels_fixed
