#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

#define T 12

__global__ void opt_2dhistoKernel(uint32_t*, size_t, size_t, uint32_t*);
__global__ void opt_32to8Kernel(uint32_t*, uint8_t*, size_t);

void opt_2dhisto(uint32_t* input, size_t height, size_t width, uint8_t* bins, uint32_t* g_bins)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */

	// working
	dim3 block(16, 16);
	dim3 grid(((INPUT_WIDTH + 128) & 0xFFFFFF80) / 16, INPUT_HEIGHT / 16);
	opt_2dhistoKernel<<<grid, block>>>(input, height, width, g_bins);

	cudaThreadSynchronize();

	opt_32to8Kernel<<<HISTO_HEIGHT * HISTO_WIDTH / 512, 512>>>(g_bins, bins, 1024);

	cudaThreadSynchronize();
}

/* Include below the implementation of any other functions you need */
/* kernel verson 1: basic */
__global__ void opt_2dhistoKernel(uint32_t *input, size_t height, size_t width, uint32_t* bins){

	/*
	// working
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if (row == 0 && col < 1024) {
		bins[col] = 0;
	}

	__syncthreads();
	if (row < height && col < width) {
		atomicAdd(&bins[input[col + row * ((INPUT_WIDTH + 128) & 0xFFFFFF80)]], 1);
	}
	*/

	/*	
	// working
	__shared__ int temp[1024];
	temp[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	int size = HISTO_HEIGHT * HISTO_WIDTH;

	while (i < size) {
		atomicAdd(&temp[input[i]], 1);
		i += offset;
	} 
	__syncthreads();

	atomicAdd(&(bins[threadIdx.x]), temp[threadIdx.x]);	
	*/

	/*
	// working but slower
	__shared__ int temp[1024];
	temp[threadIdx.x] = 0;
	temp[threadIdx.x + 256] = 0;
	temp[threadIdx.x + 512] = 0;
	temp[threadIdx.x + 768] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	int size = HISTO_HEIGHT * HISTO_WIDTH;

	while (i < size) {
		atomicAdd(&temp[input[i]], 1);
		i += offset;
	} 
	__syncthreads();

	atomicAdd(&(bins[threadIdx.x]), temp[threadIdx.x]);
	atomicAdd(&(bins[threadIdx.x + 256]), temp[threadIdx.x + 256]);
	atomicAdd(&(bins[threadIdx.x + 512]), temp[threadIdx.x + 512]);
	atomicAdd(&(bins[threadIdx.x + 768]), temp[threadIdx.x + 768]);
	*/

	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int mask = (INPUT_WIDTH + 128) & 0xFFFFFF80;

	if (row == 0 && col < 1024) {
		bins[col] = 0;
	}

	int index;
	__syncthreads();
	if (row < height && col < width) {
		index = input[col + row * mask];
		if (bins[index] < 255)
			atomicAdd(&bins[index], 1);
	}
		
}

__global__ void opt_32to8Kernel(uint32_t *input, uint8_t* output, size_t length){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	output[idx] = (uint8_t)((input[idx] < UINT8_MAX) * input[idx]) + (input[idx] >= UINT8_MAX) * UINT8_MAX;

	__syncthreads();
}

void* AllocateDevice(size_t size){
	void* ret;
	cudaMalloc(&ret, size);
	return ret;
}

void CopyToDevice(void* D_device, void* D_host, size_t size){
	cudaMemcpy(D_device, D_host, size, 
					cudaMemcpyHostToDevice);
}

void CopyFromDevice(void* D_host, void* D_device, size_t size){
	cudaMemcpy(D_host, D_device, size, 
					cudaMemcpyDeviceToHost);
}

void FreeDevice(void* D_device){
	cudaFree(D_device);
}
