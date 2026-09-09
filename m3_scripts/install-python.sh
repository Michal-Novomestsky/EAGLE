#!/bin/bash
# One-time environment setup for EAGLE on M3.
# Run from a login node (or an interactive job) BEFORE submitting train/eval jobs:
#   bash m3_scripts/install-python.sh
set -euo pipefail

# Scratch working root: a checkout of this repo (branch michal/perceiver-resampler).
DIR=/home/michaln/ml20_scratch/michaln/EAGLE
MAMBA_ROOT=~/micromamba
ENV_PATH="${DIR}/venv/eagle-py312"

cd "$DIR"

if [[ ! -x ./bin/micromamba ]]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
fi

./bin/micromamba shell init -s bash -r "$MAMBA_ROOT"
eval "$(./bin/micromamba shell hook -s bash -r "$MAMBA_ROOT")"

./bin/micromamba create -y -p "$ENV_PATH" python=3.12
micromamba activate "$ENV_PATH"

pip3 install -r requirements.txt
pip3 install "deepspeed<0.16" datasets tiktoken

python --version
