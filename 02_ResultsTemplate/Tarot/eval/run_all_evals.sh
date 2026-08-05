#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/nfs/home/ruilourenco.it/miniconda3/envs/slantKLT/lib:$LD_LIBRARY_PATH
base_dir="$(cd "$(dirname "$0")" && pwd)"

for child_dir in "$base_dir"/*; do
  [ -d "$child_dir" ] || continue
  echo "Running evaluation in $(basename "$child_dir")"
  (
    cd "$child_dir"
    export LD_LIBRARY_PATH=/nfs/home/ruilourenco.it/miniconda3/envs/slantKLT/lib:$LD_LIBRARY_PATH
    python metrics_QM.py
  )
done
