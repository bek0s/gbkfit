#!/usr/bin/env bash

. venv/bin/activate

python -m pip install twine

for whl in wheels_fixed/*.whl; do
    python -m twine check "${whl}"
done

for whl in wheels_fixed/*.whl; do
    python -m twine upload "${whl}"
done
