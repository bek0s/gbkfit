#!/usr/bin/env bash

export MACOSX_DEPLOYMENT_TARGET=10.9
export GBKFIT_BUILD_HOST=1

brew install fftw libomp

. venv/bin/activate

python -m pip install cmake

python -m pip wheel . -w wheels --no-deps

python -m pip install delocate

ls wheels

for whl in wheels/*.whl; do
    delocate-wheel -w wheels_fixed -v "$whl"
done

ls wheels_fixed
