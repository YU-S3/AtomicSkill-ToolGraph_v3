#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO/src:$REPO"
exec "${ASG_PY:-/home/yangchengyu/asg_alfworld_venv/bin/python}" -m experiments.release_control "${1:?release root}" "${2:?all or seed}"
