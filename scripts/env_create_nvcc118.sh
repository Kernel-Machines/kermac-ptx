#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# CUDA 11.8 Conda toolchain – emits .version 7.8 / .target sm_80              #
# Works on any modern Ubuntu / WSL / container that already has Conda/Mamba.  #
# Sources:  nvidia/label/cuda-11.8.0 channel docs.                           #
###############################################################################

ENV_NAME=cuda118
CUDA_LABEL="cuda-11.8.0"                         # PTX 7.8 lives here
NV_CHANNEL="nvidia/label/${CUDA_LABEL}"

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

echo ">> Cleaning any previous attempt"
conda env remove -n "$ENV_NAME" -y || true
conda clean --packages --tarballs -y

echo ">> Enforcing strict channel priority"
conda config --set channel_priority strict

echo ">> Creating toolchain environment: $ENV_NAME"
conda create -n "$ENV_NAME" -y \
      -c "$NV_CHANNEL" \
      -c conda-forge \
      cuda-nvcc cuda-cudart-dev cuda-cccl \
      gcc=11 gxx=11                      # GCC ≤11 is supported by nvcc 11.8

# echo ">> Activating environment"
# eval "$(conda shell.bash hook)"
# conda activate "$ENV_NAME"

# # Point nvcc at the GCC that lives *inside* this env
# export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
# export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

# echo ">> Smoke-testing nvcc → PTX 7.8"
# cat > t.cu <<'EOF'
# __global__ void k() {}
# EOF

# nvcc -arch=sm_80 -ptx t.cu -ccbin "$CC"

# echo -e "\nPTX header should read 7.8 / sm_80:"
# head -n12 t.ptx
