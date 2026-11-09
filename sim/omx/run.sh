#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${ISAAC_PYTHON:=$HOME/miniconda3/envs/isaacsim/bin/python}"

export PYTHONPATH="$REPO_ROOT/sim:$REPO_ROOT/tasks:$REPO_ROOT/leisaac/source/leisaac${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES

exec "$ISAAC_PYTHON" "$@"
