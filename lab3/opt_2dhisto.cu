#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"


__global__ void HistKernel(uint32_t *deviceImage, uint32_t *deviceBins32, size_t height, size_t width) {

	size_t globalTid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t numThreads = blockDim.x;
	printf("%d\n", blockIdx.x);

	// init histogram for each block
	__shared__ uint32_t partialHist[HISTO_WIDTH + 1];

	partialHist[globalTid] = 0;

	__syncthreads();



	for (size_t j = globalTid; j < height * width; j += numThreads){
		uint32_t value = deviceImage[j];

		if(partialHist[value] < UINT8_MAX) {
			atomicAdd(&partialHist[value], 1);
		}
	}

	__syncthreads();


	// sum partials 255 *

	atomicAdd(&deviceBins32[globalTid], partialHist[globalTid]);

}

__global__ void HistKernel32to8(uint32_t *deviceBins32, uint8_t *deviceBins, size_t height, size_t width) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	deviceBins[index] = (deviceBins32[index] < UINT8_MAX) ? (uint8_t) deviceBins32[index] : (uint8_t) UINT8_MAX;
}


void opt_2dhisto(uint32_t *deviceImage, uint32_t *deviceBins32, uint8_t *deviceBins, size_t height, size_t width) {
	dim3 dimGrid((height * width - 1)/HISTO_WIDTH + 1, 1, 1);
	HistKernel <<<1, HISTO_WIDTH>>> (deviceImage, deviceBins32, height, width);
	HistKernel32to8 <<<HISTO_HEIGHT, HISTO_WIDTH>>> (deviceBins32, deviceBins, height, width);
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

void ToDeviceBins32(uint32_t *deviceBins, uint32_t *input,  size_t height, size_t width) {
	int size = width * sizeof(uint32_t);
	cudaMemcpy(deviceBins, input, size, cudaMemcpyHostToDevice);

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
