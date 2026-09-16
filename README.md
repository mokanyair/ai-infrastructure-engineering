# AI Infrastructure Engineering

Hands-on projects focused on AI infrastructure architecture, compute, storage,
networking, inference, observability, automation, performance and cost.

## Project 01 — AI Inference Infrastructure Baseline & Sizing

This project establishes a reproducible CPU inference baseline for a small
pretrained language model on AWS.

### Objectives

- characterize an LLM inference workload
- understand model memory and compute requirements
- provision infrastructure using Terraform
- benchmark latency, throughput and resource utilization
- identify infrastructure bottlenecks
- compare vertical CPU scaling
- evaluate cost/performance
- establish criteria for later CPU-versus-GPU testing

### Platform

- AWS EC2
- Amazon Linux 2023
- PyTorch
- Hugging Face Transformers
- TinyLlama-1.1B-Chat-v1.0
- Terraform
- Python

### Current status

Design phases complete. Infrastructure deployment has not started.

### Project phases

1. Workload requirements
2. Model selection
3. Infrastructure sizing
4. Cost estimation and budget controls
5. Architecture design
6. Repository setup
7. Terraform implementation
8. Runtime installation
9. Baseline measurement
10. Benchmark execution
11. Bottleneck analysis
12. Vertical scaling
13. Cost/performance analysis
14. CPU-versus-GPU decision
15. Operational failure review
16. Teardown and cost reconciliation
