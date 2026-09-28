from pathlib import Path
import csv
import statistics

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "rerun-synchronized"

precisions = ["fp32", "fp16"]
batches = [1, 2, 4, 8]

summary = []

for precision in precisions:
    for batch in batches:

        benchmark_file = RESULTS / f"benchmark_{precision}_batch{batch}.csv"
        monitor_file = RESULTS / f"gpu_monitor_{precision}_batch{batch}.csv"

        # --------------------------------------------------
        # Benchmark results
        # --------------------------------------------------
        with benchmark_file.open() as f:
            bench = list(csv.DictReader(f))

        latency = [float(r["latency_seconds"]) for r in bench]
        throughput = [float(r["output_tokens_per_second"]) for r in bench]
        peak_allocated = [float(r["max_allocated_mib"]) for r in bench]

        avg_latency = statistics.mean(latency)
        avg_throughput = statistics.mean(throughput)
        benchmark_peak_vram = max(peak_allocated)

        # --------------------------------------------------
        # Synchronized GPU telemetry
        # --------------------------------------------------
        with monitor_file.open() as f:
            telemetry = list(csv.DictReader(f))

        if not telemetry:
            raise RuntimeError(f"No telemetry samples in {monitor_file}")

        gpu_util = [float(r["gpu_util_percent"]) for r in telemetry]
        power = [float(r["power_w"]) for r in telemetry]
        vram = [float(r["memory_used_mib"]) for r in telemetry]
        temp = [float(r["temperature_c"]) for r in telemetry]

        avg_gpu = statistics.mean(gpu_util)
        peak_gpu = max(gpu_util)

        avg_power = statistics.mean(power)
        peak_power = max(power)

        telemetry_peak_vram = max(vram)
        peak_temp = max(temp)

        # --------------------------------------------------
        # Efficiency / economics
        # --------------------------------------------------
        gpu_hours_per_million = (
            1_000_000 / avg_throughput / 3600
        )

        tokens_per_second_per_watt = (
            avg_throughput / avg_power
        )

        summary.append({
            "precision": precision,
            "batch": batch,
            "samples": len(telemetry),
            "latency": avg_latency,
            "throughput": avg_throughput,
            "avg_gpu": avg_gpu,
            "peak_gpu": peak_gpu,
            "avg_power": avg_power,
            "peak_power": peak_power,
            "benchmark_peak_vram": benchmark_peak_vram,
            "telemetry_peak_vram": telemetry_peak_vram,
            "peak_temp": peak_temp,
            "tokens_per_watt": tokens_per_second_per_watt,
            "gpu_hours_million": gpu_hours_per_million,
        })

# ----------------------------------------------------------
# Print consolidated table
# ----------------------------------------------------------

print()
print("=" * 155)
print("SYNCHRONIZED GPU BENCHMARK RESULTS")
print("=" * 155)

header = (
    f"{'PREC':<6}"
    f"{'BATCH':>6}"
    f"{'SAMPLES':>9}"
    f"{'LAT(s)':>10}"
    f"{'TOK/s':>11}"
    f"{'AVG GPU%':>11}"
    f"{'PEAK GPU%':>12}"
    f"{'AVG W':>10}"
    f"{'PEAK W':>10}"
    f"{'VRAM MiB':>12}"
    f"{'TOK/s/W':>11}"
    f"{'GPU-h/1M':>11}"
)

print(header)
print("-" * 155)

for r in summary:
    print(
        f"{r['precision']:<6}"
        f"{r['batch']:>6}"
        f"{r['samples']:>9}"
        f"{r['latency']:>10.3f}"
        f"{r['throughput']:>11.2f}"
        f"{r['avg_gpu']:>11.2f}"
        f"{r['peak_gpu']:>12.2f}"
        f"{r['avg_power']:>10.2f}"
        f"{r['peak_power']:>10.2f}"
        f"{r['telemetry_peak_vram']:>12.0f}"
        f"{r['tokens_per_watt']:>11.3f}"
        f"{r['gpu_hours_million']:>11.3f}"
    )

print("=" * 155)

# ----------------------------------------------------------
# Key comparisons
# ----------------------------------------------------------

def get(precision, batch):
    return next(
        r for r in summary
        if r["precision"] == precision and r["batch"] == batch
    )

fp16_b1 = get("fp16", 1)
fp16_b8 = get("fp16", 8)

fp32_b8 = get("fp32", 8)

throughput_gain_b1_b8 = (
    fp16_b8["throughput"] / fp16_b1["throughput"]
)

latency_change_b1_b8 = (
    (fp16_b8["latency"] - fp16_b1["latency"])
    / fp16_b1["latency"]
    * 100
)

fp16_vs_fp32_b8 = (
    (fp16_b8["throughput"] - fp32_b8["throughput"])
    / fp32_b8["throughput"]
    * 100
)

vram_reduction = (
    (fp32_b8["telemetry_peak_vram"] -
     fp16_b8["telemetry_peak_vram"])
    / fp32_b8["telemetry_peak_vram"]
    * 100
)

accelerator_time_reduction = (
    (fp16_b1["gpu_hours_million"] -
     fp16_b8["gpu_hours_million"])
    / fp16_b1["gpu_hours_million"]
    * 100
)

print()
print("KEY COMPARISONS")
print("-" * 70)

print(
    "FP16 B1 -> B8 throughput multiplier : "
    f"{throughput_gain_b1_b8:.2f}x"
)

print(
    "FP16 B1 -> B8 latency change        : "
    f"{latency_change_b1_b8:+.2f}%"
)

print(
    "FP16 vs FP32 throughput at B8       : "
    f"{fp16_vs_fp32_b8:+.2f}%"
)

print(
    "FP16 vs FP32 telemetry VRAM at B8   : "
    f"{vram_reduction:.2f}% reduction"
)

print(
    "FP16 B1 -> B8 accelerator-time      : "
    f"{accelerator_time_reduction:.2f}% reduction"
)

print()

# ----------------------------------------------------------
# Save consolidated CSV
# ----------------------------------------------------------

output = RESULTS / "synchronized_summary.csv"

with output.open("w", newline="") as f:

    fieldnames = [
        "precision",
        "batch_size",
        "telemetry_samples",
        "avg_latency_seconds",
        "avg_output_tokens_per_second",
        "avg_gpu_util_percent",
        "peak_gpu_util_percent",
        "avg_power_w",
        "peak_power_w",
        "benchmark_peak_allocated_mib",
        "telemetry_peak_vram_mib",
        "peak_temperature_c",
        "tokens_per_second_per_watt",
        "gpu_hours_per_1m_output_tokens",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for r in summary:
        writer.writerow({
            "precision": r["precision"],
            "batch_size": r["batch"],
            "telemetry_samples": r["samples"],
            "avg_latency_seconds": round(r["latency"], 4),
            "avg_output_tokens_per_second": round(r["throughput"], 4),
            "avg_gpu_util_percent": round(r["avg_gpu"], 2),
            "peak_gpu_util_percent": round(r["peak_gpu"], 2),
            "avg_power_w": round(r["avg_power"], 2),
            "peak_power_w": round(r["peak_power"], 2),
            "benchmark_peak_allocated_mib": round(
                r["benchmark_peak_vram"], 2
            ),
            "telemetry_peak_vram_mib": round(
                r["telemetry_peak_vram"], 2
            ),
            "peak_temperature_c": round(r["peak_temp"], 2),
            "tokens_per_second_per_watt": round(
                r["tokens_per_watt"], 4
            ),
            "gpu_hours_per_1m_output_tokens": round(
                r["gpu_hours_million"], 4
            ),
        })

print("Saved:", output)
