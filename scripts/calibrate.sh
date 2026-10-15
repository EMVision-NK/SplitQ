#!/usr/bin/env bash
# Locate source independently of the working directory.
set -e
splitq_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
family="${1:-}"
case "$family" in
  qwen2_5_vl|llava_v15) shift ;;
  *) echo "Usage: bash scripts/calibrate.sh {qwen2_5_vl|llava_v15} [calibration arguments]" >&2; exit 2 ;;
esac
export PYTHONPATH="$splitq_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -u -m splitq.calibrate --config "$splitq_root/configs/$family.yaml" "$@"
