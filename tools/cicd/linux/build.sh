#!/usr/bin/env bash

export GBKFIT_BUILD_HOST=1

yum -y install fftw3-devel

. venv/bin/activate

python -m pip install cmake

python -m pip wheel . -w wheels --no-deps

python -m pip install auditwheel

ls wheels

for whl in wheels/*.whl; do
    auditwheel repair "$whl" -w wheels_fixed
done

ls wheels_fixed
