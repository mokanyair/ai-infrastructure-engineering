#!/usr/bin/env python3

import csv
import sys
from collections import defaultdict
from statistics import mean


if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: python summarize_results.py results/benchmark_fp32.csv "
        "[results/benchmark_fp16.csv]"
    )


print(
    f"{'PRECISION':<10}"
    f"{'BATCH':>8}"
    f"{'LATENCY(s)':>14}"
    f"{'TOKENS/s':>14}"
    f"{'PEAK VRAM MiB':>16}"
)

print("-" * 62)


for filename in sys.argv[1:]:
    grouped = defaultdict(list)

    with open(filename, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            grouped[
                (row["precision"], int(row["batch_size"]))
            ].append(row)

    for (precision, batch), rows in sorted(grouped.items()):
        latency = mean(
            float(r["latency_seconds"])
            for r in rows
        )

        throughput = mean(
            float(r["output_tokens_per_second"])
            for r in rows
        )

        peak_vram = max(
            float(r["max_allocated_mib"])
            for r in rows
        )

        print(
            f"{precision:<10}"
            f"{batch:>8}"
            f"{latency:>14.3f}"
            f"{throughput:>14.3f}"
            f"{peak_vram:>16.1f}"
        )
