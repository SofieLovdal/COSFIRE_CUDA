/*This kernel governs the flow of the rotation-invariant COSFIRE
 implementation, so it retrieves a COSFIRE response for each given 
 orientation and then takes the pixelwise maximum of the responses
 as final output
 * 
 * 
 * Parameterlist contains a number of parameters, e.a the following;
 * (0)threshold, (1)sigmaratio, (2)alpha, (3)sigma0, (4)rotationStep, (5)numRotations*/

#include "COSFIRE.cu"
#include "pixelwiseMax.cu"

__global__ void rotationInvariantCOSFIRE(double * output, double * orientationResponses, 
					double * const input,
					unsigned int const numRows, unsigned int const numCols,
					double * tuples, unsigned int const numTuples,
					double * responseBuffer1, double * responseBuffer2,
					double * parameters)

{   
	/*The rotation coefficient of this thread is the same as its thread
	 * id - one rotationInvariantCOSFIRE kernel is launched for each orientation*/
	 
	int rotationCoefficient = threadIdx.x;
	int numRotations = parameters[5];
	 
	/*Achieves a COSFIRE response for each orientation. Launch a kernel 
	 * that does this for each tuple of the orientation*/
	
	/*Each thread calculates one orientation-specific response and gets a pointer to their
	 * own offset in the buffers*/ 
	double * myResponseBuffer1 = &(responseBuffer1[numTuples*numRows*numCols*threadIdx.x]);
    double * myResponseBuffer2 = &(responseBuffer2[numTuples*numRows*numCols*threadIdx.x]);
    double * myOrientationResponse = &(orientationResponses[numRows*numCols*threadIdx.x]);
	 	
	dim3 gridSize(1, 1, 1);
    dim3 blockSize(numTuples, 1, 1);
	
	
	//for rotation invariance, only rho is different so apply shiftPixels only to this
	COSFIRE_CUDA<<<gridSize, blockSize>>>(myOrientationResponse, input, numRows, numCols, tuples,
                          numTuples, myResponseBuffer1, myResponseBuffer2, parameters, rotationCoefficient);
    
    
    output[500]=1.0;
    output[502]=1.0;
    output[505]=100;                      
    /*Wait for all responses to finish*/
    __syncthreads();
	cudaDeviceSynchronize();                      
                          
     /*Kernel achieves response for a single orientation, stores this in orientationReponses.*/
     
     /*Output: Make kernel that takes the pixewise maximum in the image*/
     
    dim3 blockSize2(16, 16, 1);
    dim3 gridSize2(ceil((double)numRows/16), ceil((double)numCols/16));
    if(threadIdx.x==0) {
       pixelwiseMax<<<gridSize2, blockSize2>>>(output, orientationResponses, numRows, numCols, numRotations);                      
	}
}
