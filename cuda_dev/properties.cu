#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


int main(int argc, char** argv)
{

	int deviceCount = 0;

	cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGetDeviceCount retornou código: %d\n -> %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
		return 0;
	}

	if (deviceCount == 0)
	{
		fprintf(stdout, "Não há dispositivo compatível com CUDA\n");
	}

	else
	{
		fprintf(stdout, "Detectado %d dispositivo(s) CUDA\n", deviceCount);
	}

	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, dev);

		fprintf(stdout, "\nDevice %d: \"%s\"\n", dev, prop.name);

		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);

		printf("CUDA Driver Version / Runtime Version %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("CUDA Capability Major/Minor version number: %d.%d\n", prop.major, prop.minor);
		printf("QTD Multiprocessors: %d \n", prop.multiProcessorCount);
		printf("Total constant memory:%zu bytes\n", prop.totalConstMem);
		printf("Total shared memory per block:%zu bytes\n", prop.sharedMemPerBlock);
		printf("Shared memory per multiprocessor:%zu bytes\n", prop.sharedMemPerMultiprocessor);
		printf("Number of registers available per block:%d\n", prop.regsPerBlock);
		printf("Memory Clock Rate (KHz): %d\n",prop.memoryClockRate);
		printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
		printf("==============Thread indexing props===========\n");
		printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
		printf("Max Threads Dimensions: X = %d, Y = %d, Z = %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max Blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);

	}

	return 0;
}