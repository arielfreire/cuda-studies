#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void transpose(int* matriz, int* transposed, int size)
{
    printf("Block: %d, Thread: %d, Block Dim: %d\n", threadIdx.x, blockIdx.x, blockDim.x);
    int column = threadIdx.x;
    int row = blockIdx.x;
    transposed[row * size + column] = matriz[column * size + row];
}

int main() {

    cudaError_t cudaStatus;
    const int size = 3;
    const int bytes = size * size * sizeof(int);
    
    int matriz[size][size] = { {1,2,3}, {4,5,6}, {7,8,9} };
    int transposed[size][size] = { {0} };

    int* device_matriz = 0;
    int* device_transposed = 0;

    cudaStatus = cudaSetDevice(0);
    cudaStatus = cudaMalloc(&device_matriz, bytes);
    cudaStatus = cudaMalloc(&device_transposed, bytes);
    
    cudaStatus = cudaMemcpy(device_matriz, matriz, bytes, cudaMemcpyHostToDevice);

    transpose << <size, size >> > (device_matriz, device_transposed, size);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("transpose function failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    cudaStatus = cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(transposed, device_transposed, bytes, cudaMemcpyDeviceToHost);


    printf("Original Matrix:\n");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf(" %d ", matriz[i][j]);
        }
        printf("\n");
    }

    printf("Transposed Matrix:\n");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf(" %d ", transposed[i][j]);
        }
        printf("\n");
    }

    cudaFree(device_matriz);
    cudaFree(device_transposed);
    return 0;
}