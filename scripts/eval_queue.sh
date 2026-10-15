#!/usr/bin/env bash
# Queue only explicitly requested checkpoints and tasks.
set -e
splitq_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$splitq_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -u -m splitq.eval_queue "$@"
