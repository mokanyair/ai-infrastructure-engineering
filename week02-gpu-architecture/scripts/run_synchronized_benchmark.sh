#!/usr/bin/env bash
set -euo pipefail

PRECISION="${1:-}"
BATCH="${2:-}"

if [[ "$PRECISION" != "fp32" && "$PRECISION" != "fp16" ]]; then
    echo "Usage: $0 <fp32|fp16> <batch_size>"
    exit 1
fi

if [[ ! "$BATCH" =~ ^[0-9]+$ ]] || [[ "$BATCH" -lt 1 ]]; then
    echo "ERROR: batch_size must be a positive integer"
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/rerun-synchronized"

mkdir -p "$OUT"

TEMP_CSV="$OUT/benchmark_${PRECISION}.csv"
TEMP_JSON="$OUT/benchmark_${PRECISION}.json"

FINAL_CSV="$OUT/benchmark_${PRECISION}_batch${BATCH}.csv"
FINAL_JSON="$OUT/benchmark_${PRECISION}_batch${BATCH}.json"
FINAL_LOG="$OUT/benchmark_${PRECISION}_batch${BATCH}.log"
MONITOR_FILE="$OUT/gpu_monitor_${PRECISION}_batch${BATCH}.csv"

rm -f \
    "$TEMP_CSV" \
    "$TEMP_JSON" \
    "$FINAL_CSV" \
    "$FINAL_JSON" \
    "$FINAL_LOG" \
    "$MONITOR_FILE"

echo "=================================================="
echo "SYNCHRONIZED GPU BENCHMARK"
echo "=================================================="
echo "Precision : $PRECISION"
echo "Batch     : $BATCH"
echo
echo "Telemetry will start AFTER warmup and stop"
echo "immediately after the measured inference runs."
echo "=================================================="
echo

python "$ROOT/scripts/gpu_benchmark.py" \
    --precision "$PRECISION" \
    --batch-sizes "$BATCH" \
    --max-new-tokens 64 \
    --warmup-runs 1 \
    --measured-runs 3 \
    --output-dir "$OUT" \
    2>&1 | tee "$FINAL_LOG"

if [[ ! -f "$TEMP_CSV" ]]; then
    echo "ERROR: Missing $TEMP_CSV"
    exit 1
fi

if [[ ! -f "$TEMP_JSON" ]]; then
    echo "ERROR: Missing $TEMP_JSON"
    exit 1
fi

if [[ ! -f "$MONITOR_FILE" ]]; then
    echo "ERROR: Missing synchronized telemetry: $MONITOR_FILE"
    exit 1
fi

mv "$TEMP_CSV" "$FINAL_CSV"
mv "$TEMP_JSON" "$FINAL_JSON"

SAMPLES=$(($(wc -l < "$MONITOR_FILE") - 1))

echo
echo "=================================================="
echo "COMPLETED SUCCESSFULLY"
echo "=================================================="
echo "Precision        : $PRECISION"
echo "Batch            : $BATCH"
echo "Telemetry samples: $SAMPLES"
echo
echo "Artifacts:"
echo "  $FINAL_CSV"
echo "  $FINAL_JSON"
echo "  $FINAL_LOG"
echo "  $MONITOR_FILE"
echo "=================================================="
