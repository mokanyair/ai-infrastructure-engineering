# AI Inference Infrastructure Baseline & Sizing

## Purpose

Build and benchmark a low-cost self-hosted LLM inference environment on AWS
to understand the relationship between model characteristics and
infrastructure requirements.

## Workload

- inference only
- TinyLlama-1.1B-Chat-v1.0
- CPU execution
- synthetic prompts
- controlled input, output and concurrency

## Infrastructure

Baseline:
- m6i.xlarge
- 4 vCPU
- 16 GiB RAM

Scaling:
- m6i.2xlarge
- 8 vCPU
- 32 GiB RAM

Storage:
- 40 GiB encrypted gp3

## Key measurements

- model load time
- request latency
- input/output token counts
- tokens per second
- requests per second
- CPU utilization
- memory utilization
- process RSS
- disk I/O
- queue wait time
- error rate

## Constraints

- no training
- no fine-tuning
- no RAG
- no GPU in Week 1
- no Kubernetes
- no production data
