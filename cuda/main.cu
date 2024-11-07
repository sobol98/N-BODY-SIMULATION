/*
    * To compile this program:
    * nvcc -o hello main.cu
    *
    * To run the program:
    * ./hello
*/


#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cuda_hello() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
    printf("eeeee");
}



int main(){
    cuda_hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}



