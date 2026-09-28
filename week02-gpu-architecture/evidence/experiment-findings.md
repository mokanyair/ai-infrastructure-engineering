# Week 2 — GPU Architecture and Utilization Findings

## Engineering Question

The purpose of this experiment was not to demonstrate that a GPU is
faster than a CPU.

The primary question was:

How much useful work can be extracted from a paid accelerator, and how
do workload shape and numerical precision affect throughput, latency,
memory consumption and GPU economics?

## Platform

- AWS EC2 g4dn.xlarge
- NVIDIA Tesla T4
- 1 GPU
- 40 SMs
- ~16 GiB physical GPU memory
- PyTorch 2.14.0+cu130
- Transformers 5.17.0
- PyTorch CUDA runtime 13.0
- NVIDIA driver 595.91.07

## Workload

Model:
TinyLlama/TinyLlama-1.1B-Chat-v1.0

Revision:
fe8a4ea1ffedaf415f4da2f062534de366a451e6

Test variables:

- FP32 and FP16
- Batch sizes 1, 2, 4 and 8
- 64 maximum generated tokens
- 1 warm-up run
- 3 measured runs

## Results

| Precision | Batch | Latency (s) | Tokens/s | Peak VRAM (MiB) |
|-----------|------:|------------:|---------:|----------------:|
| FP32 | 1 | 1.339 | 47.799 | 4211.9 |
| FP32 | 2 | 1.694 | 75.545 | 4219.2 |
| FP32 | 4 | 1.865 | 137.239 | 4234.0 |
| FP32 | 8 | 2.503 | 204.536 | 4263.5 |
| FP16 | 1 | 1.398 | 45.781 | 2112.0 |
| FP16 | 2 | 1.430 | 89.511 | 2116.3 |
| FP16 | 4 | 1.447 | 176.973 | 2123.1 |
| FP16 | 8 | 1.469 | 348.488 | 2138.5 |

## Key Findings

### 1. Lower precision was not automatically faster

At batch size 1, FP16 delivered approximately 45.8 tokens/s versus
47.8 tokens/s for FP32.

Precision alone therefore did not produce a throughput improvement
for the smallest workload tested.

### 2. FP16 became increasingly valuable as batch size increased

At batch size 8:

- FP16 throughput: 348.488 tokens/s
- FP32 throughput: 204.536 tokens/s
- FP16 throughput advantage: approximately 70.4%
- FP16 latency: 1.469 seconds
- FP32 latency: 2.503 seconds
- FP16 latency reduction: approximately 41.3%

### 3. FP16 approximately halved peak model VRAM

At batch size 8:

- FP32 peak VRAM: 4263.5 MiB
- FP16 peak VRAM: 2138.5 MiB

This represents approximately 49.8% lower peak VRAM consumption.

### 4. Workload shape materially changed accelerator productivity

FP16 throughput increased from 45.781 tokens/s at batch size 1 to
348.488 tokens/s at batch size 8.

This represents approximately 7.61x throughput while measured latency
increased by only approximately 5.1%.

## GPU Economics

The same accelerator has a fixed infrastructure cost per unit of
running time. Increasing useful throughput therefore reduces the
accelerator-time required per unit of work.

Approximate GPU-hours required per one million generated tokens:

| Configuration | GPU-hours / 1M output tokens |
|---------------|------------------------------:|
| FP32 Batch 1 | 5.81 |
| FP32 Batch 8 | 1.36 |
| FP16 Batch 1 | 6.07 |
| FP16 Batch 8 | 0.80 |

Within this controlled benchmark, FP16 batch 8 required approximately
86.9% less accelerator-time per million generated tokens than FP16
batch 1.

This is not an AWS billing estimate. It is a relative accelerator
productivity metric derived from measured output-token throughput.

## Monitoring Evidence

FP32 monitor:
- 111 samples
- Peak GPU utilization: 100%
- Peak VRAM: 4439 MiB
- Peak power: 83.65 W
- Peak temperature: 48 C

FP16 monitor:
- 1218 samples
- Peak GPU utilization: 67%
- Peak VRAM: 2389 MiB
- Peak power: 62.78 W
- Peak temperature: 47 C

## Monitoring Limitation

The FP32 and FP16 monitoring windows were not equivalent.

FP32 contained 111 samples while FP16 contained 1218 samples.
Consequently, whole-file average GPU utilization, memory usage and
power consumption cannot be directly compared because the FP16
monitoring file contains substantially more idle time.

The raw monitoring data is retained for transparency, but average
power and utilization comparisons are intentionally excluded from
the primary conclusions.

A future experiment should synchronize GPU telemetry collection with
each individual benchmark run and batch size.

## Experiment Limitations

This experiment should not be generalized as a universal GPU sizing
rule.

Limitations include:

- One NVIDIA Tesla T4
- One approximately 1.1B parameter language model
- Single-GPU execution
- Batch size is not equivalent to production request concurrency
- No inference server such as vLLM or NVIDIA Triton
- No continuous production request queue
- No quantization
- No TensorRT-LLM
- No multi-GPU execution
- No NVLink or NCCL
- No production network/storage workload
- Short-duration benchmark
- Unsynchronized FP32/FP16 telemetry windows
- No modern H100/H200/B200 comparison

## Infrastructure Insight

The experiment suggests that accelerator selection alone is an
incomplete infrastructure decision.

GPU economics depends on the interaction between:

workload shape + precision + accelerator utilization + memory
efficiency + latency requirements + throughput requirements + cost.

An expensive accelerator that spends significant time underutilized
can have worse economics than a correctly sized accelerator operating
at higher productive utilization.

The infrastructure objective is therefore not simply to provision a
GPU. It is to maximize useful work per accelerator-hour while meeting
the application's latency and reliability requirements.
