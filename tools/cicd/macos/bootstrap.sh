#!/usr/bin/env bash

PYTHON_VERSIONS=(
  ["310"]="3.10.9"
)

PYTHON_VERSION="${1//./}"

brew update
#brew upgrade
brew install pyenv
pyenv install -l
export MACOSX_DEPLOYMENT_TARGET=10.9
pyenv install "${PYTHON_VERSIONS[$PYTHON_VERSION]}"
"$HOME"/.pyenv/versions/"${PYTHON_VERSIONS[$PYTHON_VERSION]}"/bin/python -m venv venv
. venv/bin/activate
python -m pip install pip -U
