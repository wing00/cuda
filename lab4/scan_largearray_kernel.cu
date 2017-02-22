#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 256

// Lab4: Host Helper Functions (allocate your own data structure...)


// Lab4: Device Functions


// Lab4: Kernel Functions
__global__ void prescanKernel(float *outArray, float *inArray, int numElements) {
	outArray[0] = 0;
	double total_sum = 0;
	for( unsigned int i = 1; i < numElements; ++i) {
	      total_sum += inArray[i-1];
	      outArray[i] = inArray[i-1] + outArray[i-1];
	  }
}


// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{

	prescanKernel<<<1 , 1>>> (outArray, inArray, numElements);


}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
