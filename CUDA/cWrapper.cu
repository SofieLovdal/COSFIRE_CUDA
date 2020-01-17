/*C-Wrapper to check the COSFIRE_CUDA algorithm flow*/

#include "COSFIRE.cu"
#include "cuda.h"
#include <stdio.h>
#include "readBin.c"

#DEFINE NUMROWS 565
#DEFINE NUMCOLS 584

main() {
	/*create input for GPU*/
	
	double *input, *tuples, *output, *responseBuffer1, *responseBuffer2;
	int numTuples=9, i;
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
	
	 double *input_on_GPU, *tuples_on_GPU, *parameters_on_GPU;
   double *DoGResponse, *maxBlurBuffer, *shiftedPixelsBuffer, *orientationResponses;
   double *output, *outMatrix;
    
   /*Allocate space on GPU for the necessary variables and buffers*/
   cudaMalloc((void**)&input_on_GPU, numRows*numCols*sizeof(double));
   cudaMalloc((void**)&tuples_on_GPU, 2*numTuples*sizeof(double));
   cudaMalloc((void**)&parameters_on_GPU, NUMPARAMS*sizeof(double));
   
   cudaMalloc((void**)&DoGResponse, numRows*numCols*sizeof(double));
   cudaMalloc((void**)&maxBlurBuffer, numRhos*numRows*numCols*sizeof(double));
   cudaMalloc((void**)&shiftedPixelsBuffer, numTuples*numRows*numCols*sizeof(double));
   cudaMalloc((void**)&orientationResponses, numOrientations*numRows*numCols*sizeof(double));
   cudaMalloc((void**)&output, numRows*numCols*sizeof(double));
   
   err = cudaGetLastError();
   if ( cudaSuccess != err ) { 
      mexPrintf("cudaCheckError() failed at cudaMalloc %s\n", cudaGetErrorString( err ) );
   }
   
   /*Copy over the input arguments to the GPU before calling kernel*/
   cudaMemcpy(input_on_GPU, input, numRows*numCols*sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(tuples_on_GPU, tuples, 2*numTuples*sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(parameters_on_GPU, parameters, NUMPARAMS*sizeof(double), cudaMemcpyHostToDevice);
   
   err = cudaGetLastError();
   if ( cudaSuccess != err ) {
      mexPrintf("cudaCheckError() failed at cudaMemcpy %s\n", cudaGetErrorString( err ) );
   }
   
   /* start timer */
   totalClock = timer();
   clock=timer();
   
   /*allocate a buffer on the GPU for the 2D DoG filter*/
   double *DoGfilter;
   int sz = ceil(sigma*3) * 2 + 1;
   cudaMalloc((void**)&DoGfilter, 100*sz*sz*sizeof(double));
   
   err = cudaGetLastError();
   if ( cudaSuccess != err ) {
      mexPrintf("cudaCheckError() failed at cudaMalloc DoGfilter %s\n", cudaGetErrorString( err ) );
   }
   
   dim3 gridSize(1, 1, 1);
   dim3 blockSize(sz, sz, 1);
   getDoG<<<gridSize, blockSize>>>(DoGfilter, sigma, sigmaRatio);
   
   err = cudaGetLastError();
   if ( cudaSuccess != err ) {
      mexPrintf("cudaCheckError() failed at getDoG %s\n", cudaGetErrorString( err ) );
   }
   cudaDeviceSynchronize();
   
   /*recover optimal blockSize from GPU architecture */
   /*If only one stream is used, kernel launches are queued and executed sequentially, so supposedly no cudadevicesynchronize is needed*/
   
   /*convolute input with DoG filter*/
   dim3 blockSize2 (32, 32, 1);
   dim3 gridSize2 (ceil((double)numRows/32), ceil((double)numCols/32));
   conv2<<<gridSize2, blockSize2>>>(DoGResponse, input_on_GPU, numRows, numCols, DoGfilter, sz, sz);
   err = cudaGetLastError();
   if ( cudaSuccess != err ) {
      mexPrintf("cudaCheckError() failed at conv2 %s\n", cudaGetErrorString( err ) );
   }
   cudaDeviceSynchronize();
   
   clock=timer()-clock;
   timings[0]=clock;
   
   clock=timer();
   /*Next: For each rho in the set of tuples, perform a maxblurring. 
    * We have the unique rhos and number of unique rhos and the actual list of rhos, 
    * so just do the maxBlurring for all unique rhos now */
   double blurSigma;
   for (i=0; i<numRhos; i++) {
   	  blurSigma = sigma0 + alpha*uniqueRhos[i];
      sz = ceil(blurSigma*3.0)*2+1;
	  dim3 blockSize3(sz, sz, 1);
	  getGaussian<<<1, blockSize3>>>(DoGfilter, blurSigma);
	  cudaDeviceSynchronize();
      maxBlur<<<gridSize2, blockSize2>>>(&maxBlurBuffer[i*numRows*numCols], DoGResponse, numRows, numCols, DoGfilter, sz, sz);
      cudaDeviceSynchronize();
   }
   
   clock=timer()-clock;
   timings[1]=clock;
   
   if ( cudaSuccess != err ) {
      mexPrintf("cudaCheckError() failed at getGaussian/maxBlur %s\n", cudaGetErrorString( err ) );
   }
   
   /*This array keeps track of which tuple can be linked to which maxBlurring-response.
   * The index denotes the tuple, and the contents of corresponding element denotes
   * the index (offset) in the maxBlurBuffer. */
   int mapping[numTuples];
   mapUniqueRhos(uniqueRhos, numRhos, mapping, tuples, numTuples);
   
   clock=timer();
   
   double rho, phi;
   for(i=0; i<numOrientations; i++) {
   	  for(j=0; j<numTuples; j++) {
	     rho = tuples[2*j];
		 phi = tuples[2*j+1] + rotationStep*i;
         shiftPixels<<<gridSize2, blockSize2>>>(&shiftedPixelsBuffer[j*numRows*numCols], &maxBlurBuffer[mapping[j]*numRows*numCols], numRows, numCols, rho, phi);
	  }
	  cudaDeviceSynchronize();
	  geometricMean<<<gridSize2, blockSize2>>>(&orientationResponses[i*numRows*numCols], shiftedPixelsBuffer, numRows, numCols, numTuples, threshold);
   }
   
   clock=timer()-clock;
   timings[2]=clock;
   
   clock=timer();
   pixelwiseMax<<<gridSize2, blockSize2>>>(output, orientationResponses, numRows, numCols, numOrientations);	   
   if ( cudaSuccess != err ) {
      mexPrintf("cudaCheckError() failed at shiftPixels/geometricMean/pixelwiseMax %s\n", cudaGetErrorString( err ) );
   }
   clock=timer()-clock;
   timings[3]=clock;
   
   /*stop timer*/
   totalClock = timer() - totalClock;
   timings[4]=totalClock;
   
   /*Create the output matrix, get a pointer to the real data in this */
   plhs[0] = mxCreateDoubleMatrix(1,numRows*numCols,mxREAL);
   outMatrix = mxGetPr(plhs[0]);
   
   double *runTimes;
   plhs[1] = mxCreateDoubleMatrix(1,5,mxREAL);
   runTimes = mxGetPr(plhs[1]);
   
   for(i=0; i<5; i++) {
	   runTimes[i]=timings[i];
   }	   
   
   /*Copy final response from GPU to CPU*/
   cudaMemcpy(outMatrix, output, numRows*numCols*sizeof(double), cudaMemcpyDeviceToHost);

   err = cudaGetLastError();
   if ( cudaSuccess != err ) {
      mexPrintf("cudaCheckError() failed at copying output from GPU to CPU %s\n", cudaGetErrorString( err ) );
   }
   
   /*Free GPU memory*/ 
   cudaFree(input_on_GPU);
   cudaFree(tuples_on_GPU);
   cudaFree(parameters_on_GPU);
   cudaFree(DoGResponse);
   cudaFree(DoGfilter);
   cudaFree(maxBlurBuffer);
   cudaFree(shiftedPixelsBuffer);
   cudaFree(orientationResponses);
   cudaFree(output);
 
}
