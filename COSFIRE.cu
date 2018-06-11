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
*/

__global__ void COSFIRE_CUDA(double * output, double * const input,
					unsigned int const numRows, unsigned int const numCols,
					double const * tuples, unsigned int const numTuples)
{	   
   /*Maximize GPU load. Sync before output merging*/
   
   /*Create DoG filter for each sigma in set S*/
   
   /*Convolve with input -- separable filters?? */
   
   /*Create Gaussian blur filter for each rho-sigma combination*/
   
   /*Convolve (max-blurring) with each corresponding response from DoG convolution*/
   
   /*Obtain final response by inspecting subresponses (array of 2D matrices
    * and their corresponding shift info is needed)*/
   
   /*current pixel*/
   const int rowIdx = blockIdx.x*blockDim.x + threadIdx.x;
   const int colIdx = blockIdx.y;
   const int sliceIdx = threadIdx.z;
   
   /*make sure we are within image*/
   if(rowIdx>=numRows) return;
	
   /*Compute the index of my element */
   const unsigned int linearIdx = rowIdx + colIdx*numRows + sliceIdx*numRows*numCols;
   
   if(linearIdx>=numRows*numCols) return;
  
   /*assign value to output*/
   output[linearIdx]=input[linearIdx];
}

