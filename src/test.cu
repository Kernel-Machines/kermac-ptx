#include <stdio.h>

#include <cute/tensor.hpp>

extern "C"
__global__
void
test_kernel() {
    // using namespace cute;

    // auto sA = make_shape(10,100);
    // print(sA);
    // printf("dogs\n");
}

int
main() {
    test_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
}