#!/usr/bin/env bash

brew install fftw

export MACOSX_DEPLOYMENT_TARGET=10.9
export GBKFIT_BUILD_OPENMP=1

. venv/bin/activate

python -m pip install cmake

python -m pip wheel . -w wheels --no-deps

python -m pip install delocate

ls wheels

for whl in wheels/*.whl; do
    python -m delocate-wheel -w wheels_fixed -v "$whl"
done

ls wheels_fixed
