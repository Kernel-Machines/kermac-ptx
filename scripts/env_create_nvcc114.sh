#!/usr/bin/env bash
set -euo pipefail

# --- 0.  housekeeping --------------------------------------------------------
# remove any stale environment and cached, half-downloaded packages
conda env remove -n cuda114 -y || true
conda clean --packages --tarballs -y

# enforce “first channel wins” so nothing leaks in from defaults
conda config --set channel_priority strict

# --- 1.  create the toolchain env -------------------------------------------
#   • CUDA 11.4.4   → PTX 8.0
#   • GCC 11.x      → passes nvcc’s version gate
#   • Headers/libs  → cuda-cudart-dev, cuda-cccl
conda create -n cuda114 -y \
      -c "nvidia/label/cuda-11.4.4" \
      -c conda-forge \
      cuda-nvcc cuda-cudart-dev cuda-cccl \
      gcc=11 gxx=11

# --- 2.  activate it ---------------------------------------------------------
# shell-agnostic activation for inside scripts
eval "$(conda shell.bash hook)"
conda activate cuda114

# point nvcc at the GCC that lives *inside* this env
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# --- 3.  smoke test ----------------------------------------------------------
cat > t.cu <<'EOF'
__global__ void k() {}
EOF

nvcc -arch=sm_80 -ptx t.cu -ccbin "$CC"

echo -e "\nPTX header:"
head -n12 t.ptx          # should print .version 8.0 / .target sm_80
