#!/usr/bin/env bash

PYTHON_VERSIONS=(
  ["37"]="cp37-cp37m"
  ["38"]="cp38-cp38"
)

PYTHON_VERSION="${1//./}"

git submodule sync --recursive
git submodule update --init --recursive

yum -y upgrade
yum clean all
rm -rf /var/cache/yum

/opt/python/"${PYTHON_VERSIONS[$PYTHON_VERSION]}"/bin/python -m venv venv
. venv/bin/activate
python -m pip install pip -U
