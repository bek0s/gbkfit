#!/usr/bin/env bash

export GBKFIT_BUILD_OPENMP=1

. venv/bin/activate

python -m pip install cmake

python -m pip wheel . -w wheels --no-deps

python -m pip install auditwheel

for whl in wheels/*.whl; do
    python -m auditwheel repair "$whl" -w wheels
done
