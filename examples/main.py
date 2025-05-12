# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import sys

# import cupy as cp

from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch, ObjectCode

code = """

extern "C"
__global__ 
void 
saxpy(
    const float a,
    const float* x,
    const float* y,
    float* out,
    size_t N
) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i=tid; i<N; i+=gridDim.x*blockDim.x) {
        out[i] = a * x[i] + y[i];
    }
}

"""

dev = Device()
dev.set_current()
s = dev.create_stream()

arch = "".join(f"{i}" for i in dev.compute_capability)

# Compile cuda to ptx
program_options = ProgramOptions(std="c++17", arch=f"sm_{arch}")
prog = Program(code, code_type="c++", options=program_options)
mod = prog.compile(
    "ptx",
    logs=sys.stdout,
)

# Compile ptx to cubin
program_options_2 = ProgramOptions(arch=f"sm_{arch}")
prog_2 = Program(mod.code.decode(), code_type="ptx", options=program_options_2)
mod_2 = prog.compile(
    "cubin",
    logs=sys.stdout
)

# Store cubin to file
print(mod_2.code)
with open('output.cubin', 'wb') as file:
    file.write(mod_2.code)

# read cubin from file
mod_3 = ObjectCode.from_cubin('output.cubin')

# access saxpy from cubin and print num_registers
print(mod_3.get_kernel("saxpy").attributes.num_regs())

# print(mod.get_kernel("saxpy<float>").attributes.num_regs())
# # run in single precision
# ker = mod.get_kernel("saxpy<float>")

# print(mod2)
# dtype = cp.float32

# # prepare input/output
# size = cp.uint64(64)
# a = dtype(10)
# rng = cp.random.default_rng()
# x = rng.random(size, dtype=dtype)
# y = rng.random(size, dtype=dtype)
# out = cp.empty_like(x)
# dev.sync()  # cupy runs on a different stream from s, so sync before accessing

# # prepare launch
# block = 32
# grid = int((size + block - 1) // block)
# config = LaunchConfig(grid=grid, block=block)
# ker_args = (a, x.data.ptr, y.data.ptr, out.data.ptr, size)

# # launch kernel on stream s
# launch(s, config, ker, *ker_args)
# s.sync()

# # check result
# assert cp.allclose(out, a * x + y)

# # let's repeat again, this time allocates our own out buffer instead of cupy's
# # run in double precision
# ker = mod.get_kernel("saxpy<double>")
# dtype = cp.float64

# # prepare input
# size = cp.uint64(128)
# a = dtype(42)
# x = rng.random(size, dtype=dtype)
# y = rng.random(size, dtype=dtype)
# dev.sync()

# # prepare output
# buf = dev.allocate(
#     size * 8,  # = dtype.itemsize
#     stream=s,
# )

# # prepare launch
# block = 64
# grid = int((size + block - 1) // block)
# config = LaunchConfig(grid=grid, block=block)
# ker_args = (a, x.data.ptr, y.data.ptr, buf, size)

# # launch kernel on stream s
# launch(s, config, ker, *ker_args)
# s.sync()

# # check result
# # we wrap output buffer as a cupy array for simplicity
# out = cp.ndarray(
#     size, dtype=dtype, memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(int(buf.handle), buf.size, buf), 0)
# )
# assert cp.allclose(out, a * x + y)

# # clean up resources that we allocate
# # cupy cleans up automatically the rest
# buf.close(s)
# s.close()

# print("done!")