#!/usr/bin/env bash

. venv/bin/activate

for whl in wheels/*.whl; do
    python -m pip install "$whl"
done

python -m pip install pytest

python -m pytest tests/
