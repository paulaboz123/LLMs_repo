#!/usr/bin/env bash
set -euo pipefail

# One-command deployment (Linux/macOS):
#   bash infra/deploy.sh
#
# Prereqs:
# - az login (or a service principal with env vars set)
# - Required env vars documented in README.md

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r infra/requirements.deploy.txt

python infra/deploy.py
