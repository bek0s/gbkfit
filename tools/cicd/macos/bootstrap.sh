#!/usr/bin/env bash

PYTHON_VERSIONS=(
  ["37"]="3.7.4"
  ["38"]="3.8.0"
)

PYTHON_VERSION="${1//./}"

git submodule sync --recursive
git submodule update --init --recursive
export MACOSX_DEPLOYMENT_TARGET=10.9
brew update
brew upgrade
brew install pyenv

pyenv install "${PYTHON_VERSIONS[$PYTHON_VERSION]}"
"$HOME"/.pyenv/versions/"${PYTHON_VERSIONS[$PYTHON_VERSION]}"/bin/python -m venv venv
. venv/bin/activate
python -m pip install pip -U
