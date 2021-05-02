#!/usr/bin/env bash

PYTHON_VERSIONS=(
  ["39"]="3.9.0"
)

PYTHON_VERSION="${1//./}"

git submodule sync --recursive
git submodule update --init --recursive

brew update
brew upgrade
brew install pyenv

export MACOSX_DEPLOYMENT_TARGET=10.9
pyenv install "${PYTHON_VERSIONS[$PYTHON_VERSION]}"
"$HOME"/.pyenv/versions/"${PYTHON_VERSIONS[$PYTHON_VERSION]}"/bin/python -m venv venv
. venv/bin/activate
python -m pip install pip -U
