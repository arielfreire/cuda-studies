
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int error(int* device_a);

__global__ void identity(int* device_a, int size)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    printf("Block: (%d, %d), Thread: (%d, %d), Block Dim: (%d, %d)\n", blockIdx.x, blockIdx.y,  threadIdx.x, threadIdx.y, blockDim.x, blockDim.y);
    printf("x: %d, y: %d\n", x, y);

    if(x == y)
    {
        device_a[x*size + y] = 1;
    }
}

int main()
{
    cudaError_t cudaStatus;
    const int matrixSize = 5;
    int a[matrixSize][matrixSize] = { { 0 } };
    const size_t bytes = matrixSize * matrixSize * sizeof(int);
  
    int* device_a;
   
    // Seleciona o dispositivo
    cudaStatus = cudaSetDevice(0);

    //Aloca memoria na GPU
    cudaStatus = cudaMalloc(&device_a, bytes);

    cudaStatus = cudaMemcpy(device_a, a, bytes, cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice falhou!");
        return error(device_a);
    }

    dim3 grid(1, 1);
    dim3 block(matrixSize, matrixSize, 1);
    identity<<<grid, block >>>(device_a, matrixSize);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("add function failed: %s\n", cudaGetErrorString(cudaStatus));
        return error(device_a);
    }

    cudaStatus = cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(a, device_a, bytes, cudaMemcpyDeviceToHost);
    cudaStatus = cudaDeviceReset();


    printf("Identity Matrix:\n");
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            printf(" %d ", a[i][j]);
        }
        printf("\n");
    }
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    cudaFree(device_a);
    return 0;
}

int error(int* device_a) {
    cudaFree(device_a);
    return 1;
}

