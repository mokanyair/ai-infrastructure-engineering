#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-results/gpu_monitor.csv}"
INTERVAL="${2:-1}"

mkdir -p "$(dirname "$OUTPUT")"

echo "timestamp,name,pstate,memory_total_mib,memory_used_mib,gpu_util_percent,memory_util_percent,temperature_c,power_w" > "$OUTPUT"

echo "GPU monitoring started: $OUTPUT"
echo "Press Ctrl+C to stop."

while true; do
    nvidia-smi \
      --query-gpu=timestamp,name,pstate,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
      --format=csv,noheader,nounits >> "$OUTPUT"

    sleep "$INTERVAL"
done
