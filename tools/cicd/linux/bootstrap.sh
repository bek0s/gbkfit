#!/usr/bin/env bash

PYTHON_VERSIONS=(
  ["312"]="cp312-cp312"
)

PYTHON_VERSION="${1//./}"

dnf -y upgrade
dnf clean all
rm -rf /var/cache/dnf

/opt/python/"${PYTHON_VERSIONS[$PYTHON_VERSION]}"/bin/python -m venv venv
. venv/bin/activate
python -m pip install pip -U
