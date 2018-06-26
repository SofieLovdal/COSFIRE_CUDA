/*C-Wrapper to check the COSFIRE_CUDA algorithm flow*/

#include "COSFIRE.cu"
#include "cuda.h"
#include <stdio.h>

main() {
	/*create input for GPU*/
	
	double *input, *tuples, *output, *responseBuffer1, *responseBuffer2;
	int numRows=20, numCols=20, numTuples=1, i;
	double sigmaratio=0.5, threshold=0.5;
	double *input_on_GPU, *tuples_on_GPU, *output_on_GPU;
	cudaError err;
	
	double sigma0 = 0.5;
	double alpha = 0.1167;
	
	input = (double*)malloc(numRows*numCols*sizeof(double));
	tuples = (double*)malloc(3*numTuples*sizeof(double));
	output = (double*)malloc(numRows*numCols*sizeof(double));
	for(i=0; i<numRows*numCols; i++) {
		input[i]=1;
	}	
	
	for(i=0; i<numTuples*3; i++) {
		tuples[i]=2.0;
	}	

	
	/*Allocate space on GPU for the necessary variables
    * Works after resetting GPU by logging in and out*/
   cudaMalloc((void**)&input_on_GPU, numRows*numCols*sizeof(double));
   cudaMalloc((void**)&tuples_on_GPU, 3*numTuples*sizeof(double));
   cudaMalloc((void**)&output_on_GPU, numRows*numCols*sizeof(double));
   cudaMalloc((void**)&responseBuffer1, numTuples*numRows*numCols*sizeof(double));
   cudaMalloc((void**)&responseBuffer2, numTuples*numRows*numCols*sizeof(double));
   
   /*Copy over some input arguments to the GPU before calling kernel*/
   cudaMemcpy(input_on_GPU, input, numRows*numCols*sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(tuples_on_GPU, tuples, 3*numTuples*sizeof(double), cudaMemcpyHostToDevice);
   err = cudaGetLastError();
   if ( cudaSuccess != err )
   {
      printf("cudaCheckError() failed at cudaMemcpy %s\n", cudaGetErrorString( err ) );
   }	
		
   printf("launching COSFIRE kernel\n");		
	
   dim3 gridSize(1, 1, 1);
   dim3 blockSize(numTuples, 1, 1);
   /*Make kernel call with the GPU variables*/
   COSFIRE_CUDA<<<gridSize, blockSize>>>(output_on_GPU, input_on_GPU, numRows, numCols, tuples_on_GPU,
                          numTuples, responseBuffer1, responseBuffer2, threshold, sigmaratio, alpha, sigma0);

    err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
       printf("cudaCheckError() failed at COSFIRE_CUDA call %s\n", cudaGetErrorString( err ) );
    }
	printf("we get here\n");

   //hopefully the final response (output) is now copied into the mex output buffer
   cudaMemcpy(output, output_on_GPU, numRows*numCols*sizeof(double), cudaMemcpyDeviceToHost);
   //cudaMemcpy(output, responseBuffer1, numRows*numCols*sizeof(double), cudaMemcpyDeviceToHost);
   
   double sum=0;
   for(i=0; i<numRows*numCols; i++) {
	   printf("output[%d]: %f\n", i, output[i]);
	   sum+=output[i];
   }
   printf("sum output = %f \n", sum);
   
   cudaFree(input_on_GPU);
   cudaFree(tuples_on_GPU);
   cudaFree(output_on_GPU);
   cudaFree(responseBuffer1);
   cudaFree(responseBuffer2);
   free(input);
   free(output);
	
	return 0;
}	
