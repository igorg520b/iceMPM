#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>


__global__ void cuda_hello(){
    printf("Hello World from GPU!\n\n");
}

void test_cuda()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if(error_id) std::cout << "cudaGetDeviceCount returs error " << error_id << '\n';
    std::cout << "CUDA devices " << deviceCount << '\n';

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    printf("Device \"%s\"\n", deviceProp.name);
    printf("Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

    cuda_hello<<<1,1>>>();
}
