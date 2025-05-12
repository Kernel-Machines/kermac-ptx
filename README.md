# kermac-ptx
Repository of kernels for kermac base repo.

Meant for absorbing the long compilation times from Nvidia's cute/cutlass to emit ptx. This allows [kermac](https://github.com/Kernel-Machines/kermac) to download the ptx artifact directly and jit the ptx in the cuda.core python module. This entirely removes the Cpython, Pytorch CUDAExtension nonsense.

`scripts/env_create_nvcc118.sh` creates a conda environment that makes a workable nvcc with 11.8. 

`scripts/cmake_build_nvcc118.sh` executes cmake in this environment to emit ptx with version 7.8 targeting sm_80