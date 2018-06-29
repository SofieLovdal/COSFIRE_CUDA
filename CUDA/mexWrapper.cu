/*A mex wrapper function in order to be able to use dynamic parallelism,
 * which is not supported directly via CudaKernel objects in Matlab.
 * This function starts up the algorithm flow of the CUDA kernels.*/

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include "rotationInvariantCOSFIRE.cu"
#include "cuda.h"
#include <stdio.h>

/*Number of parameters passed to function as well as maximum expected rho-value*/
#define NUMPARAMS 7
#define MAXRHO 100

/*returns the amount of unique elements in an array given that the array
 * is sorted in ascending order. Adapt to small arrays/segfault*/
int numUniqueRhos(double *rhoList, int length, double *mapping) {
   
   int i;
   double value = tupleList[0];
   mapping[0] = 0;
   int sum = 1;
   
   for(i=1; i<length, i++) {
	   if (value != tupleList[i] ) {
		   mapping[i]=sum;
		   sum++;
	   } else {
		   mapping[i]=sum;
	   }	   	   
	   value = tupleList[i];
   }
   
   if (value != tupleList[length-2]) {
	   sum+=2;
	   mapping[length]
   
   return sum;
}   	   

/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
   //The following is absolutely necessary to pass: input (preprocessed image) with dims, tuples, numTuples, sigmaratio, threshold and other parameters
   //Buffers and similar can be created here, e.g. output, responseBuffer
   double *input, *tuples, *output, *maxBlurBuffer, *responseBuffer2, *outMatrix;
   int numRows, numCols, numTuples;
   double *input_on_GPU, *tuples_on_GPU, *parameters_on_GPU, *orientationResponses;
   double *parameters;
   double *DoGfilter, *DoGResponse;
   cudaError err;  
   int i, j;	
   
   if(nrhs != 7) {
      mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs", "Seven inputs required.");
   }
   if(nlhs != 1) {
      mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs", "One output required.");
   }
   
   int gpuInit = mxInitGPU();
   
   input = mxGetPr(prhs[0]);
   numRows = mxGetScalar(prhs[1]);
   numCols = mxGetScalar(prhs[2]);
   tuples = mxGetPr(prhs[3]);
   numTuples = mxGetScalar(prhs[4]);
   parameters = mxGetPr(prhs[5]);
   
   double sigma = parameters[0];
   double sigmaRatio = parameters[1];
   double threshold = parameters[2];
   double alpha = parameters[3];
   double sigma0 = parameters[4];
   double rotationStep = parameters[5]
   int numOrientations = parameters[6];
   
   /*This array keeps track of which tuple can be linked to which maxBlurring-response.
    * The index denotes the tuple, and the contents of corresponding element denotes
    * the index (offset) in the maxBlurBuffer.*/
   int tupleToMaxBlurResponseMap[numTuples];
   
   mexPrintf("numOrientations: %d \n", numOrientations);
   
   int numRhos = numUniqueRhos(tuples, numTuples, tupleToMaxBlurResponseMap);
   
   /*Allocate space on GPU for the necessary variables */
   cudaMalloc((void**)&input_on_GPU, numRows*numCols*sizeof(double));
   cudaMalloc((void**)&tuples_on_GPU, 3*numTuples*sizeof(double));
   cudaMalloc((void**)&DoGResponse, numRows*numCols*sizeof(double));
   cudaMalloc((void**)&maxBlurBuffer, numRhos*numRows*numCols*sizeof(double));
   
   cudaMalloc((void**)&responseBuffer2, numOrientations*numTuples*numRows*numCols*sizeof(double));
   cudaMalloc((void**)&orientationResponses, numOrientations*numRows*numCols*sizeof(double));
   cudaMalloc((void**)&parameters_on_GPU, 6*sizeof(double));
   cudaMalloc((void**)&output, numRows*numCols*sizeof(double));
   
   err = cudaGetLastError();
   if ( cudaSuccess != err )
   {
      mexPrintf("cudaCheckError() failed at cudaMemcpy %s\n", cudaGetErrorString( err ) );
   }
   
   /*Copy over some input arguments to the GPU before calling kernel*/
   cudaMemcpy(input_on_GPU, input, numRows*numCols*sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(tuples_on_GPU, tuples, 2*numTuples*sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(parameters_on_GPU, parameters, 7*sizeof(double), cudaMemcpyHostToDevice);
   
   err = cudaGetLastError();
   if ( cudaSuccess != err )
   {
      mexPrintf("cudaCheckError() failed at cudaMemcpy %s\n", cudaGetErrorString( err ) );
   }
   	
   /* create the output matrix, get a pointer to the real data in this */
   plhs[0] = mxCreateDoubleMatrix(1,numRows*numCols,mxREAL);
   outMatrix = mxGetPr(plhs[0]);	
	
	
   /*The following lines of code governs the algorithm flow*/
   
   /*allocate a buffer on the GPU for the 2D DoG filter*/
   int sz = ceil(mySigma*3) * 2 + 1;
   cudaMalloc((void**)&DoGfilter, sz*sz*sizeof(double));
   
   dim3 gridSize(1, 1, 1);
   dim3 blockSize(sz, sz, 1);
   getDoG<<<gridSize, blockSize>>>(DoGfilter, sigma, sigmaRatio);
   
   //recover optimal blockSize from GPU architecture
   /*convolute input with DoG filter*/
   dim3 blockSize2 (16, 16, 1);
   dim3 gridSize2 (ceil((double)numRows/16), ceil((double)numCols/16));
   conv2<<<gridSize2, blockSize2>>>(DoGResponse, input, numRows, numCols, DoGfilter, sz, sz);
   
   /*Next: For each rho in the set of tuples, perform a maxblurring.*/
   /*This assumes symmetric filter, perhaps better to retrieve unique rhos in another way
    * Then, if you have unique number of rhos, you could allocate buffers based on that.*/
   for (i=0; i<numRhos; i++) {
   	  double blurSigma = sigma0 + alpha*myRho; //CHANGE SIZE OF FILTER + NO NORMALIZATION OF VALUES
      sz = ceil(blurSigma*3.0)*2+1;
	  dim3 blockSize3(sz, sz, 1);
	  getGaussian<<<1, blockSize3>>>(DoGfilter, blurSigma);
      maxBlur<<<gridSize2, blockSize2>>>(myResponse2, myResponse1, numRows, numCols, DoGfilter, sz, sz);
   }   
	   
	
	
   dim3 gridSize(1, 1, 1);
   dim3 blockSize(numOrientations, 1, 1);
   /*Make kernel call with the GPU variables
    * move management out here 
    * DoG
    * for for blurring
    * for for orientation
    * pixelwise max*/
   rotationInvariantCOSFIRE<<<gridSize, blockSize>>>(output, orientationResponses, input_on_GPU, numRows, numCols, tuples_on_GPU,
                          numTuples, responseBuffer1, responseBuffer2, parameters_on_GPU);
   
    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
       mexPrintf("cudaCheckError() failed at COSFIRE_CUDA call %s\n", cudaGetErrorString( err ) );
    }

   /*Copy final response from GPU to CPU*/
   cudaMemcpy(outMatrix, output, numRows*numCols*sizeof(double), cudaMemcpyDeviceToHost);
   
    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
       mexPrintf("cudaCheckError() failed at copying output from GPU to CPU %s\n", cudaGetErrorString( err ) );
    }
    
   cudaFree(input_on_GPU);
   cudaFree(tuples_on_GPU);
   cudaFree(output);
   cudaFree(responseBuffer1);
   cudaFree(responseBuffer2);
   cudaFree(orientationResponses);
   cudaFree(parameters);
   
}
