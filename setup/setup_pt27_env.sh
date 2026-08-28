#!/usr/bin/env bash
# Setup script for vla-adapter-pt27 conda environment
# PyTorch 2.7 + Triton 3.3+ on GH200 (aarch64) with cuda-compat/12.8
#
# Usage: Run on a compute node (interactive or SLURM) with GPU access:
#   srun --account=<account> -p <partition> -N 1 --gpus-per-node=1 --cpus-per-task=12 --mem=32g -t 01:00:00 --pty bash
#   bash setup_pt27_env.sh

set -euo pipefail

CONDA_SH="${CONDA_SH:?Set CONDA_SH to the conda.sh path for your installation}"
CONDA_ENV="${CONDA_ENV:-vla-adapter-pt27}"
REPO_DIR="${REPO_DIR:-$(pwd)}"

echo "============================================"
echo "  Setting up ${CONDA_ENV}"
echo "============================================"
echo ""

# Optionally load site-specific CUDA compatibility and CUPTI paths.
if [ -n "${CUDA_COMPAT_MODULE:-}" ]; then
    module load "$CUDA_COMPAT_MODULE"
fi
if [ -n "${CUPTI_LIB_DIR:-}" ]; then
    export LD_LIBRARY_PATH="${CUPTI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Step 1: Create conda environment
echo ""
echo "[Step 1/4] Creating conda environment..."
source "${CONDA_SH}"

if conda env list | grep -q "${CONDA_ENV}"; then
    echo "  Environment '${CONDA_ENV}' already exists. Activating..."
else
    conda create -n "${CONDA_ENV}" python=3.11 -y
    echo "  Created environment '${CONDA_ENV}' with Python 3.11"
fi
conda activate "${CONDA_ENV}"
echo "  Python: $(python -V)"

# Step 2: Install PyTorch 2.7 (cu128 aarch64)
echo ""
echo "[Step 2/4] Installing PyTorch 2.7 (cu128)..."
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Step 3: Install VLA-Adapter dependencies
echo ""
echo "[Step 3/4] Installing VLA-Adapter dependencies..."
cd "${REPO_DIR}"
pip install -e ".[dev]"

# Step 4: Fix triton version (VLA-Adapter deps may install pytorch-triton 3.5+ which conflicts)
echo ""
echo "[Step 4/4] Pinning triton==3.3.0 (required by PyTorch 2.7 inductor)..."
pip uninstall triton pytorch-triton pytorch-triton-rocm -y 2>/dev/null || true
pip install triton==3.3.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
python -c "import triton; print(f'  Triton: {triton.__version__}')"

echo ""
echo "============================================"
echo "  Environment setup complete!"
echo "============================================"
echo ""
echo "Summary:"
echo "  Conda env: ${CONDA_ENV}"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"
python -c "import triton; print(f'  Triton: {triton.__version__}')"
echo ""
echo "Next steps:"
echo "  1. (optional) pip install flash-attn --no-build-isolation"
echo "  2. Run a CALVIN eval: sbatch slurm/slurm_calvin_eval.slurm"
