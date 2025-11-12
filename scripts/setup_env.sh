#!/bin/bash
# Reference installation on Linux (Python 3.10, CUDA 12.1, torch 2.5.1, taichi 1.7.3; used on H100 GPUs).
# Usage: bash scripts/setup_env.sh [venv_dir]      (needs: a CUDA 12.x toolkit with nvcc for the two source builds, gcc <= 12)
set -e
VENV=${1:-.venv}
python3.10 -m venv "$VENV" || python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install "numpy<2"
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install pymeshlab || true      # hole closing of open scans (scripts/data/make_watertight.py)
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import taichi as ti; print('taichi', ti.__version__)"
# pytorch3d (k-nearest neighbours / marching cubes for the differentiable tasks; the code falls back to torch / scikit-image without it)
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;8.6;8.9;9.0"}
python -c "import pytorch3d" 2>/dev/null || pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
# nvdiffrast (inverse rendering, Table 4 only; NVIDIA Source Code License, research use)
python -c "import nvdiffrast" 2>/dev/null || pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast@v0.4.0"
python scripts/smoke_test.py
