#!/usr/bin/env bash
# Use explicit model/data paths and the active Python environment.
set -e
splitq_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$splitq_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -u -m splitq.evaluate "$@"
