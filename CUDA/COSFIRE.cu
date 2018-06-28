/*
* CUDA kernel for effective implementation of the COSFIRE algorithm.
* 
* Takes a preprocessed input image and a set of tuples as input (preprocessing
* of image and configuration of filter is done in Matlab).
* The kernel performs the following steps in parallel:
* 
* 1. Generates DoG filters corresponding to each unique sigma
* and convolves this with the input image
* 
* 2. Generates Gaussian blur filter for each unique (sigma, rho)-combination
* in the set of tuples and performs dilation (max-blurring) on each corresponding
* response from (1.)
* 
* 3.Output the response image by weighted geometric mean after shifting 
* each subresponse according to (rho, theta).
* 
* Sofie Lovdal 5.6.2018
* 
* For now, the implementation allocates the necessary big chunks of 
* memory on host side and passes it to the algorithm governing kernel.
*/
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"

#include "getDoG.cu"
#include "convolution.cu"
#include "maxBlur.cu"
#include "shiftPixels.cu"
#include "geometricMean.cu"
#include "getGaussian.cu"

/*some maximum size for some buffers*/
__constant__ int MAXSIZE=900;

/* Parameters in input: (0)threshold, (1)sigmaratio, (2)alpha, (3)sigma0,
 * (4)rotationStep, (5)numRotations*/

__global__ void COSFIRE_CUDA(double * output, double * const input,
					unsigned int const numRows, unsigned int const numCols,
					double * tuples, unsigned int const numTuples,
					double * responseBuffer1, double * responseBuffer2,
					double * parameters, int rotationNumber)
{	   
    
    double * myTuple = &(tuples[3*threadIdx.x]);
    double * myResponse1 = &(responseBuffer1[numRows*numCols*threadIdx.x]);
    double * myResponse2 = &(responseBuffer2[numRows*numCols*threadIdx.x]);
    
    double threshold = parameters[0];
    double sigmaratio = parameters[1];
    double alpha = parameters[2];
    double sigma0 = parameters[3];
    double rotationStep = parameters[4];
    
    double mySigma = myTuple[0];
    double myRho = myTuple[1];
    double myPhi = myTuple[2]+rotationStep*rotationNumber;
    
    double * DoGfilter;
    DoGfilter = (double*)malloc(MAXSIZE*sizeof(double));
	

	int sz = ceil(mySigma*3) * 2 + 1; //related to calculating suitable block size for getDoG kernel launch
	dim3 gridSize (1);
	dim3 blockSize (sz, sz, 1);
    getDoG<<<1, blockSize>>>(DoGfilter, mySigma, sigmaratio); //launch one grid with blocksize sz. Every tuple-thread does this - dynamic parallelism.


	dim3 blockSize2 (16, 16, 1);
    dim3 gridSize2 (ceil((double)numRows/16), ceil((double)numCols/16));
	conv2<<<gridSize2, blockSize2>>>(myResponse1, input, numRows, numCols, DoGfilter, sz, sz);
    
    
	double blurSigma = sigma0 + alpha*myRho; //CHANGE SIZE OF FILTER + NO NORMALIZATION OF VALUES
	sz = ceil(blurSigma*3.0)*2+1;
	dim3 blockSize3(sz, sz, 1);
	getGaussian<<<1, blockSize3>>>(DoGfilter, blurSigma);
    maxBlur<<<gridSize2, blockSize2>>>(myResponse2, myResponse1, numRows, numCols, DoGfilter, sz, sz);


    shiftPixels<<<gridSize2, blockSize2>>>(myResponse1, myResponse2, numRows, numCols, myRho, myPhi);
   
   
    __syncthreads();
	cudaDeviceSynchronize();
   
   
   if(threadIdx.x == 0) {
	   geometricMean<<<gridSize2, blockSize2>>>(output, responseBuffer1, numRows, numCols, numTuples, threshold);
   }
   
}

