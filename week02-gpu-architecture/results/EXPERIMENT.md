# Week 2 GPU Architecture Experiment

## Objective

Determine how workload shape and numerical precision affect GPU
utilization, throughput, latency, VRAM consumption, power consumption
and accelerator economics.

## Infrastructure

- Cloud: AWS
- Instance: g4dn.xlarge
- GPU: NVIDIA Tesla T4
- GPU count: 1
- GPU SMs: 40
- VRAM: ~16 GiB
- vCPU: 4
- System RAM: 16 GiB

## Runtime

- PyTorch: 2.14.0+cu130
- Transformers: 5.17.0
- PyTorch CUDA runtime: 13.0
- NVIDIA Driver: 595.91.07
- Driver CUDA compatibility: 13.2

## Model

TinyLlama/TinyLlama-1.1B-Chat-v1.0

Revision:

fe8a4ea1ffedaf415f4da2f062534de366a451e6

## Variables

Precision:
- FP32
- FP16

Batch:
- 1
- 2
- 4
- 8

Generated tokens:
- 64 maximum

Measured runs:
- 3 per configuration

Warm-up:
- 1 run

## Metrics

- Latency
- Output tokens/second
- Requests/second
- GPU utilization
- VRAM
- Power
- Temperature

## Important Limitations

This is a controlled infrastructure experiment, not a production
inference benchmark.

Limitations include:

- Single NVIDIA T4 GPU
- Single ~1.1B parameter model
- Batch size is not equivalent to production request concurrency
- No vLLM or NVIDIA Triton inference server
- No quantization
- No TensorRT-LLM
- No multi-GPU execution
- No NVLink/NCCL
- No production network or storage workload
- Short-duration power and thermal observations
- Results should not be generalized to modern H100/H200/B200 systems

The experiment is intended to study infrastructure behavior and
accelerator utilization rather than establish universal model
performance rankings.
