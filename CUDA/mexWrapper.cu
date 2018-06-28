/*A mex wrapper function in order to be able to use dynamic parallelism,
 * which is not supported directly via CudaKernel objects in Matlab.
 * This function starts up the algorithm flow of the CUDA kernels.*/

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include "rotationInvariantCOSFIRE.cu"
#include "cuda.h"
#include <stdio.h>

/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
   //The following is absolutely necessary to pass: input (preprocessed image) with dims, tuples, numTuples, sigmaratio, threshold and other parameters
   //Buffers and similar can be created here, e.g. output, responseBuffer
   double *input, *tuples, *output, *responseBuffer1, *responseBuffer2, *outMatrix;
   int numRows, numCols, numTuples;
   double *input_on_GPU, *tuples_on_GPU, *orientationResponses;
   double *parameters;
   cudaError err;  
 
   
   if(nrhs != 7) {
      mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs", "Seven inputs required.");
   }
   if(nlhs != 1) {
      mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs", "One output required.");
   }
   
   
   int gpuInit = mxInitGPU();
   //mexPrintf("gpuInit: %d \n", gpuInit);
   
   input = mxGetPr(prhs[0]);
   numRows = mxGetScalar(prhs[1]);
   numCols = mxGetScalar(prhs[2]);
   tuples = mxGetPr(prhs[3]);
   numTuples = mxGetScalar(prhs[4]);
   parameters = mxGetPr(prhs[6]);
   
   int numOrientations = parameters[5];
   
   /*Allocate space on GPU for the necessary variables */
   cudaMalloc((void**)&input_on_GPU, numRows*numCols*sizeof(double));
   cudaMalloc((void**)&tuples_on_GPU, 3*numTuples*sizeof(double));
   cudaMalloc((void**)&output, numRows*numCols*sizeof(double));
   cudaMalloc((void**)&responseBuffer1, numOrientations*numTuples*numRows*numCols*sizeof(double));
   cudaMalloc((void**)&responseBuffer2, numOrientations*numTuples*numRows*numCols*sizeof(double));
   cudaMalloc((void**)&orientationResponses, numOrientations*numRows*numCols*sizeof(double));
   
   /*Copy over some input arguments to the GPU before calling kernel*/
   cudaMemcpy(input_on_GPU, input, numRows*numCols*sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(tuples_on_GPU, tuples, 3*numTuples*sizeof(double), cudaMemcpyHostToDevice);
   
   err = cudaGetLastError();
   if ( cudaSuccess != err )
   {
      mexPrintf("cudaCheckError() failed at cudaMemcpy %s\n", cudaGetErrorString( err ) );
   }
   	
   /* create the output matrix, get a pointer to the real data in this */
   plhs[0] = mxCreateDoubleMatrix(1,numRows*numCols,mxREAL);
   outMatrix = mxGetPr(plhs[0]);	

   dim3 gridSize(1, 1, 1);
   dim3 blockSize(numOrientations, 1, 1);
   /*Make kernel call with the GPU variables*/
   rotationInvariantCOSFIRE<<<gridSize, blockSize>>>(output, orientationResponses, input_on_GPU, numRows, numCols, tuples_on_GPU,
                          numTuples, responseBuffer1, responseBuffer2, parameters);
   
    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
       mexPrintf("cudaCheckError() failed at COSFIRE_CUDA call %s\n", cudaGetErrorString( err ) );
    }

   /*Copy final response from GPU to CPU*/
   cudaMemcpy(outMatrix, output, numRows*numCols*sizeof(double), cudaMemcpyDeviceToHost);
   
   
   cudaFree(input_on_GPU);
   cudaFree(tuples_on_GPU);
   cudaFree(output);
   cudaFree(responseBuffer1);
   cudaFree(responseBuffer2);
}
