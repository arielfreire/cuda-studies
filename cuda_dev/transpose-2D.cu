#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void transpose2D(int* matriz, int* transposed, int size)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    printf("Block: (%d, %d), Thread: (%d, %d), Block Dim: (%d, %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, blockDim.x, blockDim.y);
    printf("x: %d, y: %d\n", x, y);
    transposed[x * size + y] = matriz[y * size + x];
}

int main() {

    cudaError_t cudaStatus;
    const int size = 3;
    const int bytes = size * size * sizeof(int);

    int matriz[size][size] = { {1,2,3}, {4,5,6}, {7,8,9} };
    int transposed[size][size] = { {0} };

    int* device_matriz;
    int* device_transposed;

    cudaStatus = cudaSetDevice(0);
    cudaStatus = cudaMalloc(&device_matriz, bytes);
    cudaStatus = cudaMalloc(&device_transposed, bytes);

    cudaStatus = cudaMemcpy(device_matriz, matriz, bytes, cudaMemcpyHostToDevice);
    dim3 grid(1, 1);
    dim3 block(size, size, 1);
    transpose2D << <grid, block >> > (device_matriz, device_transposed, size);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("transpose function failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    cudaStatus = cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(transposed, device_transposed, bytes, cudaMemcpyDeviceToHost);


    printf("Matriz original:\n");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf(" %d ", matriz[i][j]);
        }
        printf("\n");
    }

    printf("Matriz transposta:\n");
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
