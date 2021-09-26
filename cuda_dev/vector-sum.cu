
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int error(int* device_a, int* device_b, int* device_c);

__global__ void add(int *c, const int *a, const int *b)
{
    printf("Block: %d, Thread: %d, Block Dim: %d\n", threadIdx.x, blockIdx.x, blockDim.x);
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{   
    cudaError_t cudaStatus;
   
    const int arraySize = 7;
    const int a[arraySize] = { 1, 2, 3, 5, 7, 11, 13 };
    const int b[arraySize] = { 1, 2, 3, 5, 8, 13, 21 };
    int c[arraySize] = { 0 };

    int* device_a = 0;
    int* device_b = 0;
    int* device_c = 0;
    
    // Seleciona o dispositivo
    cudaStatus = cudaSetDevice(0);

    //Aloca memoria na GPU
    cudaStatus = cudaMalloc((void**)&device_a, arraySize * sizeof(int));
    cudaStatus = cudaMalloc((void**)&device_b, arraySize * sizeof(int));
    cudaStatus = cudaMalloc((void**)&device_c, arraySize * sizeof(int));

    cudaStatus = cudaMemcpy(device_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(device_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice falhou!");
        return error(device_a, device_b, device_c);
    }
   
    add<<<1, arraySize>>>(device_c, device_a, device_b);
   
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("add function failed: %s\n", cudaGetErrorString(cudaStatus));
        return error(device_a, device_b, device_c);
    }
    cudaStatus = cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(c, device_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    printf("{ 1, 2, 3, 5, 7, 11, 13 } + { 1, 2, 3, 5, 8, 13, 21 } = {%d, %d, %d, %d, %d, %d, %d}\n",
        c[0], c[1], c[2], c[3], c[4], c[5], c[6]);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    return 0;
}

int error(int* device_a, int* device_b, int* device_c) {
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    return 1;
}

