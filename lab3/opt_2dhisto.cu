#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"


__global__ void HistKernel(uint32_t *deviceImage, uint8_t *deviceBins, size_t height, size_t width) {

	for(size_t i = 0; i < height; i++) {
		for(size_t j = 0; j < width; j++) {
			const uint32_t value = deviceImage[i * width + j];

			if (deviceBins[value] < UINT8_MAX) {
				deviceBins[value]++;
			}
		}
	}
}

void opt_2dhisto(uint32_t *deviceImage, uint8_t *deviceBins, size_t height, size_t width) {

	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(1, 1, 1);

	HistKernel <<<dimGrid, dimBlock>>> (deviceImage, deviceBins, height, width);
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
