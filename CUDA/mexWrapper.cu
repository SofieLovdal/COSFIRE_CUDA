/*A mex wrapper function in order to be able to use dynamic parallelism,
 * which is not supported directly via CudaKernel objects in Matlab.
 * This function starts up the algorithm flow of the CUDA kernels.*/

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

/*Number of parameters passed to function as well as maximum expected rho-value*/
#define NUMPARAMS 8

/*Input: Array of the unique Rhos in the tuples
 * Output: An array of the same length as the number of tuples, associating each tuple with 
 * the index of the corresponding rho in the array of uniqueRhos*/
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


/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
   /*The input of the program*/
   double *input, *tuples, *parameters, *uniqueRhos;
   int numRows, numCols, numTuples, i, j;
   cudaError err;
    
   if(nrhs != 7 || nlhs!=1) {
      mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs || nlhs", "Seven inputs and one output required");
   }
   
   int gpuInit = mxInitGPU();
   
   /*Parse input. Tuples is a set of (rho, phi), sigma is same for all*/
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
    
   /*Allocate space on GPU for the necessary variables */
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
   	
   /* create the output matrix, get a pointer to the real data in this */
   plhs[0] = mxCreateDoubleMatrix(1,numRows*numCols,mxREAL);
   outMatrix = mxGetPr(plhs[0]);	
	
	
   /*The following lines of code governs the algorithm flow*/
   
   /*allocate a buffer on the GPU for the 2D DoG filter*/
   double *DoGfilter;
   int sz = ceil(sigma*3) * 2 + 1;
   cudaMalloc((void**)&DoGfilter, 100*sz*sz*sizeof(double)); //allocate one buffer that is large enough for all DoGfilters/Gaussians for the maxBlurring. Maybe call it filterBufferinstead
   
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
   //recover optimal blockSize from GPU architecture
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
   /*Next: For each rho in the set of tuples, perform a maxblurring. 
    * We have the unique rhos and number of unique rhos and the actual list of rhos, 
    * so just do the maxBlurring for all unique rhos now */
   double blurSigma;
   for (i=0; i<numRhos; i++) {
   	  blurSigma = sigma0 + alpha*uniqueRhos[i]; //CHANGE SIZE OF FILTER + NO NORMALIZATION OF VALUES
      sz = ceil(blurSigma*3.0)*2+1;
	  dim3 blockSize3(sz, sz, 1);
	  getGaussian<<<1, blockSize3>>>(DoGfilter, blurSigma);
	  cudaDeviceSynchronize();
      maxBlur<<<gridSize2, blockSize2>>>(&maxBlurBuffer[i*numRows*numCols], DoGResponse, numRows, numCols, DoGfilter, sz, sz);
      cudaDeviceSynchronize();
   }
    if ( cudaSuccess != err ) {
       mexPrintf("cudaCheckError() failed at getGaussian/maxBlur %s\n", cudaGetErrorString( err ) );
    }
   
    /*This array keeps track of which tuple can be linked to which maxBlurring-response.
    * The index denotes the tuple, and the contents of corresponding element denotes
    * the index (offset) in the maxBlurBuffer. */
   int mapping[numTuples];
   mapUniqueRhos(uniqueRhos, numRhos, mapping, tuples, numTuples);
   
   double rho, phi;
   for(i=0; i<numOrientations; i++) {
	   for(j=0; j<numTuples; j++) { //as input argument, give the pointer to the corresponding max-blurred output!
		   rho = tuples[2*j];
		   phi = tuples[2*j+1] + rotationStep*i; //phi is dependant on orientation currently considered
		   shiftPixels<<<gridSize2, blockSize2>>>(&shiftedPixelsBuffer[j*numRows*numCols], &maxBlurBuffer[mapping[j]*numRows*numCols], numRows, numCols, rho, phi);
		}
		cudaDeviceSynchronize();
		geometricMean<<<gridSize2, blockSize2>>>(&orientationResponses[i*numRows*numCols], shiftedPixelsBuffer, numRows, numCols, numTuples, threshold);
		cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();
	pixelwiseMax<<<gridSize2, blockSize2>>>(output, orientationResponses, numRows, numCols, numOrientations);	   
	if ( cudaSuccess != err ) {
       mexPrintf("cudaCheckError() failed at shiftPixels/geometricMean/pixelwiseMax %s\n", cudaGetErrorString( err ) );
    }      
	   
   /*FORALL rotation-directions {
    *  FOR EACH rho-phi combination (tuple) {
    *    shiftPixels ->> insert in buffer of size imageSize*numTuples
    *  }
    *  geometricMean ->> obtain output of one rotation-direction
    * }
    * END : maximum of all orientation responses
   
  
   /*Copy final response from GPU to CPU*/
   cudaDeviceSynchronize();
   cudaMemcpy(outMatrix, output, numRows*numCols*sizeof(double), cudaMemcpyDeviceToHost);
   
    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
       mexPrintf("cudaCheckError() failed at copying output from GPU to CPU %s\n", cudaGetErrorString( err ) );
    }
    
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
