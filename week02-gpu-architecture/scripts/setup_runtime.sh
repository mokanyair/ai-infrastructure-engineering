#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing Week 2 runtime ==="

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip numactl pciutils

python3 -m venv "$HOME/ai-infra-gpu-venv"
source "$HOME/ai-infra-gpu-venv/bin/activate"

python -m pip install --upgrade pip

# PyTorch PyPI package brings the required runtime dependencies.
# The NVIDIA driver remains supplied by the AWS DLAMI.
python -m pip install \
  torch \
  transformers \
  accelerate \
  safetensors \
  psutil

echo
echo "=== Runtime validation ==="

python - <<'PY'
import torch
import transformers

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA runtime:", torch.version.cuda)

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access CUDA")

print("GPU:", torch.cuda.get_device_name(0))
print("GPU count:", torch.cuda.device_count())

p = torch.cuda.get_device_properties(0)
print("VRAM GiB:", round(p.total_memory / 1024**3, 2))
print("SM count:", p.multi_processor_count)
PY

echo
echo "Runtime ready."
