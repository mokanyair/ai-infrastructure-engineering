# ADR-002 — M6i Compute Family

## Status
Accepted for Week 1 baseline.

## Decision
Use m6i.xlarge and m6i.2xlarge.

## Rationale
Non-burstable x86 instances avoid CPU-credit effects and allow a controlled
vertical scaling comparison.

## Limitation
CPU and memory increase together, so the experiment measures combined
vertical scaling.
