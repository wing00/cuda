#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 1024

// Lab4: Host Helper Functions (allocate your own data structure...)

int nearestPowerOf2(int input) {
	float count = 0;
	input -= 1;
	if (input == 0) {return 0;}
	while (input > 0) {
		input >>= 1;
		count++;
	}
	return exp2(count);
}

bool isPowerOf2(int input) {
	return (input & (input - 1)) == 0;
}

// Lab4: Device Functions



// Lab4: Kernel Functions
__global__ void upKernel(float *outArray, float *inArray, int numElements) {


	__shared__ float sharedArray[BLOCK_SIZE];
	sharedArray[threadIdx.x] = inArray[threadIdx.x];


	for(size_t i = 1; i < (numElements>>1); i <<= 1) {
		size_t index = 2 * i * ( threadIdx.x  + 1) - 1;

		if(index < numElements) {
			sharedArray[index] += sharedArray[index - i];
		}
		__syncthreads();
	}
	outArray[threadIdx.x] = sharedArray[threadIdx.x];

}

__global__ void downKernel(float *outArray, int numElements) {

	__shared__ float sharedArray[BLOCK_SIZE];

	sharedArray[threadIdx.x] = outArray[threadIdx.x];

	if(threadIdx.x == 0) {sharedArray[numElements -1] = 0;}
	__syncthreads();

	for(size_t i = numElements >> 1; i > 0; i>>= 1) {
		size_t index = 2 * i * ( threadIdx.x  + 1) - 1;
		//printf("%d %lu %lu %lu %0.2f %0.2f\n",threadIdx.x, i,   index, index - i, sharedArray[index], sharedArray[index - 1]);

		if(index < numElements) {
			float temp = sharedArray[index - i];
			sharedArray[index - i] = sharedArray[index];
			sharedArray[index] += temp;
		}
		__syncthreads();
	}

	outArray[threadIdx.x] = sharedArray[threadIdx.x];
}


// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{

	if(numElements <= BLOCK_SIZE && isPowerOf2(numElements)) {
		upKernel<<<1, numElements>>> (outArray, inArray, numElements);
		downKernel<<<1, numElements>>>(outArray, numElements);
	}
	else {
		upKernel<<<1, BLOCK_SIZE>>> (outArray, inArray, numElements);
		downKernel<<<1, BLOCK_SIZE>>>(outArray, numElements);
	}
	cudaThreadSynchronize();


}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
