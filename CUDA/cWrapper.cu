/*C-Wrapper to check the COSFIRE_CUDA algorithm flow*/

#include "COSFIRE.cu"
#include "cuda.h"
#include <stdio.h>

main() {
	/*create input for GPU*/
	
	double *input, *tuples, *output, *responseBuffer1, *responseBuffer2;
	int numRows=20, numCols=20, numTuples=1, i;
	double *input_on_GPU, *tuples_on_GPU, *output_on_GPU;
	cudaError err;
	
	double sigma0 = 0.5;
	double alpha = 0.1167;
	double sigmaRatio = 0.5;
    double threshold = 0.0;
	double numRotations = 12.0;
	double rotationStep = numRotations/3.14;
	
	double * parameters = {sigmaRatio, threshold, alpha, sigma0, rotationStep, numRotations};
	
	input = (double*)malloc(numRows*numCols*sizeof(double));
	tuples = (double*)malloc(3*numTuples*sizeof(double));
	output = (double*)malloc(numRows*numCols*sizeof(double));
	
	for(i=0; i<numRows; i++) {
		for(j=0; j<numCols; j++)
		if(i==j || i==j+1 || i==j-1) {
			input[i*numCols + j]=1.0;
		} else {
			input[i*numCols + j]=0.0;
		]		
	}	
	
	for(i=0; i<numTuples*3; i+=3) {
		tuples[i]=2.4;
		tuples[i+1]=2*i;
		tuples[i+2]=1.57+(i%2)*3.14;
	}	

	
	/*Allocate space on GPU for the necessary variables */
   cudaMalloc((void**)&input_on_GPU, numRows*numCols*sizeof(double));
   cudaMalloc((void**)&tuples_on_GPU, 3*numTuples*sizeof(double));
   cudaMalloc((void**)&output_on_GPU, numRows*numCols*sizeof(double));
   cudaMalloc((void**)&responseBuffer1, numTuples*numRows*numCols*sizeof(double));
   cudaMalloc((void**)&responseBuffer2, numTuples*numRows*numCols*sizeof(double));
   
   /*Copy over input arguments to the GPU before calling kernel*/
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

   cudaMemcpy(output, output_on_GPU, numRows*numCols*sizeof(double), cudaMemcpyDeviceToHost);
   
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
