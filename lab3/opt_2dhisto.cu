#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"


__global__ void HistKernel(uint32_t *deviceImage, uint32_t *deviceBins32, size_t height, size_t width) {

	size_t globalTid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t numThreads = blockDim.x * gridDim.x;


	// init histogram for each block
	__shared__ uint32_t partialHist[HISTO_WIDTH + 1];

	partialHist[threadIdx.x] = 0;
	__syncthreads();

	for (size_t j = globalTid; j < height * width; j += numThreads) {
		uint32_t value = deviceImage[j];

		if (partialHist[value] < UINT8_MAX) {
			atomicAdd(&partialHist[value], 1);
		}
	}
	__syncthreads();

	// sum partials
	if (deviceBins32[threadIdx.x] < UINT8_MAX) {
		atomicAdd(&deviceBins32[threadIdx.x], partialHist[threadIdx.x]);
	}
}

__global__ void HistKernel32to8(uint32_t *deviceBins32, uint8_t *deviceBins) {
	// convert int32 to int8; overloaded __nv_min function
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	deviceBins[index] = (uint8_t) min(deviceBins32[index], UINT8_MAX);
}


void opt_2dhisto(uint32_t *deviceImage, uint32_t *deviceBins32, uint8_t *deviceBins, size_t height, size_t width) {
	cudaMemset(deviceBins32, 0, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t)); //zeros

	// Occupancy calculator: 8 multiprocessors * 2 blocks

	HistKernel <<<16, HISTO_WIDTH>>> (deviceImage, deviceBins32, height, width);
	HistKernel32to8 <<<HISTO_HEIGHT, HISTO_WIDTH>>> (deviceBins32, deviceBins);
	cudaThreadSynchronize();

}

uint32_t *AllocateDeviceImage(size_t height, size_t width) {
	uint32_t *deviceImage;
	int size = height * width * sizeof(uint32_t);

	cudaMalloc((void**)&deviceImage, size);
	return deviceImage;
}

uint8_t *AllocateDeviceBins(size_t height, size_t width) {
	uint8_t *deviceBins;
    int size = height * width * sizeof(uint8_t);

    cudaMalloc((void**)&deviceBins, size);
    return deviceBins;
}

void FreeDeviceImage(uint32_t *deviceImage) {
	cudaFree(deviceImage);
	deviceImage = NULL;
}

void FreeDeviceBins(uint8_t *deviceBins) {
	cudaFree(deviceBins);
	deviceBins = NULL;
}

void ToDeviceImage(uint32_t *deviceImage, uint32_t *input[],  size_t height, size_t width) {
	int size = width * sizeof(uint32_t);
	for (int i = 0; i < height; i++) {
		cudaMemcpy(deviceImage + i * width, input[i], size, cudaMemcpyHostToDevice);
	}
}

void ToDeviceBins(uint8_t *deviceBins, uint8_t *hostBins, size_t height, size_t width) {
	int size = height * width * sizeof(uint8_t);
    cudaMemcpy(deviceBins, hostBins, size, cudaMemcpyHostToDevice);
}

void FromDeviceImage(uint32_t *hostImage, uint32_t *deviceImage, size_t height, size_t width) {
	int size = height * width * sizeof(uint32_t);
    cudaMemcpy(hostImage, deviceImage, size, cudaMemcpyDeviceToHost);
}

void FromDeviceBins(uint8_t *hostBins, uint8_t *deviceBins, size_t height, size_t width) {
	int size = height * width * sizeof(uint8_t);
    cudaMemcpy(hostBins, deviceBins, size, cudaMemcpyDeviceToHost);
}
