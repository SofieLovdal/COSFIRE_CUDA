/*
* This is a mex wrapper for an effective implementation of the COSFIRE algorithm in CUDA.
* 
* Takes a preprocessed input image and a set of tuples as input (preprocessing
* of image and configuration of filter is done in Matlab), together with parameters.
* 
* The algorithm performs the following steps:
* 
* 1. Generates a DoG filter corresponding to the sigma of each tuple 
* and convolves this with the input image
* 
* 2. Generates a Gaussian blur filter for each unique rho in the set of 
* tuples and for each filter then performs dilation (max-blurring) on 
* the DoG response from step (1.)
* 
* 3. Shift the responses from (2.) according to the (rho, phi)-values
* of the corresponding tuple.
* 
* 4. Output is obtained by calculating the pixelwise geometric mean of 
* the responses from (3.).
* 
* 5. Rotation-invariance is obtained by performing steps 3-4 with a rotated
* input pattern (modified phi). The final output is the pixelwise maximum
* of each considered orientation. 
* 
* Sofie Lovdal 4.7.2018
* 
*/
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include "cuda.h"
#include <stdio.h>
#include "getDoG.cu"
#include "maxBlur.cu"
#include "convolution.cu"
#include "geometricMean.cu"
#include "pixelwiseMax.cu"
#include "shiftPixels.cu"
#include "getGaussian.cu"
#include <sys/time.h>

/*Number of parameters passed to function*/
#define NUMPARAMS 8

/*Timer for speedup measurements*/
static double timer(void)
{
  struct timeval tm;
  gettimeofday (&tm, NULL);
  return tm.tv_sec + tm.tv_usec/1000000.0;
}

/*Maps each tuple to the index of the corresponding rho in the list of unique rhos*/
void mapUniqueRhos(double *uniqueRhos, int numRhos, int *mapping, double *tuples, int numTuples) {
   
	int i, j;
	double value;
	double epsilon = 0.0001;
	for(i=0; i<numRhos; i++) {
		value = uniqueRhos[i];
		for(j=0; j<numTuples*2; j+=2) {
			if(abs(tuples[j]-value)<epsilon) {
				mapping[j/2]=i;
			}	
		}
	}				
}

/*The gateway function for Matlab*/
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
   /*The input of the program*/
   double *input, *tuples, *parameters, *uniqueRhos;
   int numRows, numCols, numTuples, i, j;
   cudaError err;
   double clock, totalClock;
   double timings[5]; //0=getDoG, 1=maxBlur, 2=shift and geometric mean, 3=max, 4=total
     
   if(nrhs != 7 || nlhs!=2) {
      mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs || nlhs", "Seven inputs and one output required");
   }
   
   int gpuInit = mxInitGPU();
   
   /*Parse input. Tuples is a set of (rho, phi)-values, sigma is same for all tuples*/
   input = mxGetPr(prhs[0]);
   numRows = mxGetScalar(prhs[1]);
   numCols = mxGetScalar(prhs[2]);
   tuples = mxGetPr(prhs[3]);
   numTuples = mxGetScalar(prhs[4]);
   parameters = mxGetPr(prhs[5]);
   uniqueRhos = mxGetPr(prhs[6]);
   
   double sigma = parameters[0];
   double sigmaRatio = parameters[1];
   double threshold = parameters[2];
   double alpha = parameters[3];
   double sigma0 = parameters[4];
   double rotationStep = parameters[5];
   int numOrientations = parameters[6];
   double numRhos = parameters[7];
   
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
