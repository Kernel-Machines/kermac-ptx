# kermac-ptx
Repository of kernels for kermac base repo.

Meant for absorbing the long compilation times from Nvidia's cute/cutlass to emit ptx. This allows [kermac](https://github.com/Kernel-Machines/kermac) to download the ptx artifact directly and jit the ptx in the cuda.core python module. This entirely removes the Cpython, Pytorch CUDAExtension nonsense.

`scripts/nvcc-11.4.sh` creates a conda environment that makes a workable nvcc with 11.4. This emits ptx with version 7.4 targeting sm_80