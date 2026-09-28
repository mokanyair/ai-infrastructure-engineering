#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-results/gpu_inventory}"
mkdir -p "$OUT"

echo "Collecting GPU infrastructure evidence..."

hostname > "$OUT/hostname.txt"
uname -a > "$OUT/uname.txt"

lscpu > "$OUT/lscpu.txt"

if command -v numactl >/dev/null 2>&1; then
    numactl --hardware > "$OUT/numa.txt"
fi

lspci > "$OUT/lspci.txt"
lspci | grep -i -E 'nvidia|vga|3d' > "$OUT/gpu_pcie.txt" || true

nvidia-smi > "$OUT/nvidia-smi.txt"
nvidia-smi -q > "$OUT/nvidia-smi-query.txt"
nvidia-smi -L > "$OUT/nvidia-smi-list.txt"
nvidia-smi topo -m > "$OUT/nvidia-topology.txt"

nvidia-smi \
  --query-gpu=timestamp,name,uuid,driver_version,pci.bus_id,pstate,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit \
  --format=csv \
  > "$OUT/gpu_baseline.csv"

echo
echo "=== GPU ==="
nvidia-smi -L

echo
echo "=== TOPOLOGY ==="
nvidia-smi topo -m

echo
echo "Evidence saved under: $OUT"
