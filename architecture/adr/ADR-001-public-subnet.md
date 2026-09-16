# ADR-001 — Public Subnet for Lab Egress

## Status
Accepted for Week 1 lab.

## Decision
Use a public subnet with no inbound security-group rules.

## Rationale
Avoid NAT Gateway fixed cost while permitting outbound HTTPS for package and
model downloads.

## Trade-off
This is a cost-conscious lab architecture, not the preferred production
network pattern.
