/*A mex wrapper function in order to be able to use dynamic parallelism,
 * which is not supported directly via CudaKernel objects in Matlab.
 * This function starts up the algorithm flow of the CUDA kernels.*/

#include "mex.h"
#include "COSFIRE.cu"
#include "cuda.h"

/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
   //The following is absolutely necessary to pass: input (preprocessed image) with dims, tuples, numTuples, sigmaratio, threshold and other parameters
   //Buffers and similar can be created here, e.g. output, responseBuffer
   double *input, *tuples, *output, *responseBuffer1, *responseBuffer2;
   int numRows, numCols, numTuples;
   double sigmaratio, threshold;
   
   if(nrhs != 7) {
      mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs", "Seven inputs required.");
   }
   if(nlhs != 1) {
      mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs", "One output required.");
   }
   
   input = mxGetDoubles(prhs[0]);
   numRows = mxGetScalar(prhs[1]);
   numCols = mxGetScalar(prhs[2]);
   tuples = mxGetDoubles(prhs[3]);
   numTuples = mxGetScalar(prhs[[4]);
   sigmaratio = mxGetScalar(prhs[[5]);
   threshold = mxGetScalar(prhs[[6]);
   
   cudaMalloc((void**)&output, numRows*numCols*sizeof(double));
   cudaMalloc((void**)&responseBuffer1, numTuples*numRows*numCols*sizeof(double));
   cudaMalloc((void**)&responseBuffer2, numTuples*numRows*numCols*sizeof(double));
	
   /* create the output matrix */
   plhs[0] = mxCreateDoubleMatrix(1,nRows*nCols,mxREAL);
   //Use the mxGetDoubles function to assign the outMatrix argument to plhs[0]
   
   /* get a pointer to the real data in the output matrix */
   outMatrix = mxGetDoubles(plhs[0]);
	
	
   COSFIRE_CUDA<<<1, numTuples>>>(output, input, numRows, numCols, tuples, numTuples, responseBuffer1, responseBuffer2, threshold, sigmaratio);
   
   //hopefully the final response (output) is now copied into the mex output buffer
   cudaMemcpy(&outMatrix, output, numRows*numCols*sizeof(double), cudaMemcpyDeviceToHost);
}
