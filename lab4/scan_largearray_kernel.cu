#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 1024
#define OFFSET(n) ((n) >> LOG_NUM_BANKS)
// Lab4: Host Helper Functions (allocate your own data structure...)

int nextPowerOf2(float input) {
	return exp2(ceil(log2(input - 1)));
}

bool isPowerOf2(int input) {
	return (input & (input - 1)) == 0;
}

float *setBlockSums(int size) {
	float* blockSums = NULL;

	CUDA_SAFE_CALL( cudaMalloc( (void**) &blockSums,  sizeof( float) * size) );
	CUDA_SAFE_CALL( cudaMemset( blockSums, 0, sizeof( float) * size) );

	return blockSums;
}

// Lab4: Device Functions



// Lab4: Kernel Functions
__global__ void singleKernel(float *outArray, float *inArray, int numElements) {
	__shared__ float sharedArray[BLOCK_SIZE + OFFSET(BLOCK_SIZE)];


	sharedArray[threadIdx.x + OFFSET(threadIdx.x)] = (threadIdx.x < numElements) ? inArray[threadIdx.x] : 0;
	__syncthreads();

	for(size_t i = 1; i < BLOCK_SIZE; i <<= 1) {
		size_t index = 2 * i * ( threadIdx.x  + 1) - 1;

		if(index < BLOCK_SIZE) {
			sharedArray[index + OFFSET(index)] += sharedArray[index - i  + OFFSET(index - i)];
		}
		__syncthreads();
	}

	if(threadIdx.x == 0) {sharedArray[BLOCK_SIZE - 1 + OFFSET(BLOCK_SIZE - 1)] = 0;}

	for(size_t i = BLOCK_SIZE >> 1; i > 0; i >>= 1) {
		size_t index = 2 * i * ( threadIdx.x  + 1) - 1;

		if(index < BLOCK_SIZE) {
			float temp = sharedArray[index - i + + OFFSET(index - i)];
			sharedArray[index - i + OFFSET(index - i)] = sharedArray[index + OFFSET(index)];
			sharedArray[index + OFFSET(index)] += temp;
		}
		__syncthreads();
	}

	outArray[threadIdx.x] = sharedArray[threadIdx.x + OFFSET(threadIdx.x)];
}


__global__ void upKernel(float *outArray, float *inArray, float *blockSums, int numElements) {
	size_t globalThread = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float sharedArray[BLOCK_SIZE + OFFSET(BLOCK_SIZE)];
	sharedArray[threadIdx.x + OFFSET(threadIdx.x)] = (globalThread < numElements) ? inArray[globalThread] : 0;
	__syncthreads();

	for(size_t i = 1; i < BLOCK_SIZE; i <<= 1) {
		size_t index = 2 * i * ( threadIdx.x  + 1) - 1;

		if(index < BLOCK_SIZE) {
			sharedArray[index + OFFSET(index)] += sharedArray[index - i + OFFSET(index - i)];
		}
		__syncthreads();
	}

	if(threadIdx.x == 0) {
		blockSums[blockIdx.x] = sharedArray[BLOCK_SIZE - 1  + OFFSET(BLOCK_SIZE - 1)];
		sharedArray[BLOCK_SIZE - 1 +  + OFFSET(BLOCK_SIZE - 1)] = 0;
	}

	for(size_t i = BLOCK_SIZE >> 1; i > 0; i >>= 1) {
		size_t index = 2 * i * ( threadIdx.x  + 1) - 1;

		if(index < BLOCK_SIZE) {
			float temp = sharedArray[index - i + OFFSET(index - i)];
			sharedArray[index - i + OFFSET(index - i)] = sharedArray[index + OFFSET(index)];
			sharedArray[index + OFFSET(index)] += temp;
		}
		__syncthreads();
	}

	outArray[globalThread] = sharedArray[threadIdx.x  + OFFSET(threadIdx.x)];
}

__global__ void addKernel(float *outArray, float *blockSums) {
	size_t globalThread = blockIdx.x * blockDim.x + threadIdx.x;
	outArray[globalThread] += blockSums[blockIdx.x];
}


// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, float *blockSums, float *blockSumsSums, int numElements)
{

	if(numElements <= BLOCK_SIZE) {
		singleKernel <<<1, BLOCK_SIZE>>> (outArray, inArray, numElements);
	} else if (numElements <= BLOCK_SIZE * BLOCK_SIZE) {
		size_t numBlocks = (numElements - 1) / BLOCK_SIZE + 1;

		upKernel<<<numBlocks, BLOCK_SIZE>>> (outArray, inArray, blockSums, numElements);
		singleKernel<<<1, BLOCK_SIZE>>> (blockSums, blockSums, numBlocks);
		addKernel<<<numBlocks, BLOCK_SIZE>>> (outArray, blockSums);

	} else {
		size_t numBlocks = (numElements - 1)/ BLOCK_SIZE + 1;
		size_t numBlockSums = (numBlocks - 1) / BLOCK_SIZE + 1;

		upKernel<<<numBlocks, BLOCK_SIZE>>> (outArray, inArray, blockSums, numElements);
		upKernel<<<numBlockSums, BLOCK_SIZE>>> (blockSums, blockSums, blockSumsSums, numBlocks);

		singleKernel<<<1, BLOCK_SIZE>>> (blockSumsSums, blockSumsSums, numBlocks);

		addKernel<<<numBlockSums, BLOCK_SIZE>>> (blockSums, blockSumsSums);
		addKernel<<<numBlocks, BLOCK_SIZE>>> (outArray, blockSums);
	}

}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
