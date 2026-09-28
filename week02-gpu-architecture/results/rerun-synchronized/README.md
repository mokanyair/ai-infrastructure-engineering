# Week 2 — Synchronized GPU Telemetry Rerun

## Purpose

The original Week 2 GPU benchmark correctly measured inference latency and
throughput, but the FP32 and FP16 telemetry monitoring windows contained
different amounts of idle time.

This made the original average GPU utilization and average power values
unsuitable for direct comparison.

The experiment was therefore repeated with telemetry synchronized to the
measured inference window.

## Platform

- GPU: NVIDIA Tesla T4
- GPU memory: 15,360 MiB reported by nvidia-smi
- SM count: 40
- Driver: 595.91.07
- PyTorch: 2.14.0+cu130
- Transformers: 5.17.0
- PyTorch CUDA runtime: 13.0
- Precision tested: FP32 and FP16
- Batch sizes: 1, 2, 4, 8
- Measured runs per configuration: 3
- Output tokens per request: 64

## Corrected Methodology

Each precision/batch configuration was executed separately.

For each configuration:

1. Model and tokenizer were loaded.
2. Warmup completed.
3. GPU telemetry monitoring started.
4. Three measured inference runs executed.
5. GPU telemetry monitoring stopped.
6. Benchmark and telemetry results were stored in configuration-specific files.

This removed the unequal idle-time problem in the original telemetry.

## Results

| Precision | Batch | Latency (s) | Tokens/s | Avg GPU % | Avg Power W | Peak VRAM MiB | Tokens/s/W | GPU-hours/1M tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 1 | 1.334 | 47.98 | 76.50 | 57.73 | 4355 | 0.831 | 5.789 |
| FP32 | 2 | 1.679 | 76.25 | 80.71 | 58.78 | 4363 | 1.297 | 3.643 |
| FP32 | 4 | 1.851 | 138.28 | 81.00 | 67.33 | 4395 | 2.054 | 2.009 |
| FP32 | 8 | 2.464 | 207.81 | 84.33 | 62.58 | 4439 | 3.321 | 1.337 |
| FP16 | 1 | 1.372 | 46.66 | 42.33 | 55.76 | 2347 | 0.837 | 5.954 |
| FP16 | 2 | 1.420 | 90.13 | 46.00 | 52.66 | 2349 | 1.711 | 3.082 |
| FP16 | 4 | 1.461 | 175.22 | 45.83 | 52.48 | 2357 | 3.339 | 1.585 |
| FP16 | 8 | 1.474 | 347.38 | 48.50 | 50.54 | 2389 | 6.873 | 0.800 |

## Key Findings

### 1. Workload shape strongly affected accelerator productivity

Increasing FP16 batch size from 1 to 8 increased measured throughput from
46.66 to 347.38 output tokens/second, a 7.45x increase.

Average measured latency increased from 1.372 to 1.474 seconds, approximately
7.44%.

### 2. Accelerator-time per unit of work fell substantially

At the measured throughput:

- FP16 batch 1: approximately 5.95 GPU-hours per 1M output tokens
- FP16 batch 8: approximately 0.80 GPU-hours per 1M output tokens

This represents an approximately 86.57% reduction in extrapolated
accelerator-time per million output tokens.

This is an accelerator-productivity metric, not a complete AWS billing
estimate.

### 3. Precision mattered more as workload density increased

At batch 1, FP16 did not outperform FP32 in throughput.

At batch 8, FP16 delivered approximately 67.16% higher measured throughput
than FP32.

Observed peak telemetry VRAM at batch 8 decreased from 4439 MiB with FP32 to
2389 MiB with FP16, approximately 46.18%.

### 4. GPU utilization alone is not an efficiency metric

FP16 produced substantially higher throughput at larger batch sizes even
though sampled nvidia-smi GPU utilization percentages were lower than FP32.

Accelerator efficiency should therefore be evaluated using multiple metrics,
including throughput, latency, memory consumption, power, and accelerator-time
per unit of useful work.

## Telemetry Limitation

The corrected telemetry windows are synchronized with measured inference, but
nvidia-smi sampling is approximately one second while individual benchmark
runs are short.

Only 6-9 telemetry samples were captured per configuration.

Average utilization and power are therefore treated as short-window sampled
observations rather than high-resolution power profiling measurements.

Latency, throughput, VRAM, and accelerator-time per million output tokens are
the strongest evidence from this experiment.

## Engineering Conclusion

GPU infrastructure efficiency depends on more than accelerator selection.

Workload shape and numerical precision materially changed the amount of useful
work extracted from the same NVIDIA T4.

The experiment demonstrates why GPU sizing decisions should consider workload
density, latency requirements, throughput, memory footprint, power, and useful
work per accelerator-hour rather than relying on GPU utilization percentage
alone.
