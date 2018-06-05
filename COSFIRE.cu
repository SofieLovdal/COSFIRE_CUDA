/*
* CUDA kernel for effective implementation of the COSFIRE algorithm.
* 
* Takes a preprocessed input image and a set of tuples as input (preprocessing
* of image and configuration of filter is done in Matlab).
* The kernel performs the following steps in parallel:
* 
* 1. Generates k filters corresponding to each unique (sigma, rho)-combination
* in the set of tuples as a convolution of their corresponding DoG filter 
* and Gaussian blur filter.
* 
* 2.Convolve the input image with each of the k filters from (1.)
* 
* 3.Output the response image after shifting each subresponse according to
* (rho, theta).
* 
* Sofie Lovdal 5.6.2018
*/

__global__ void COSFIRE_CUDA(double * output, double * const input,
					unsigned int const numRows, unsigned int const numCols,
					double const * tuples, unsigned int const numTuples)
{	
	
   /*Find number of unique rho-sigma combinations*/
   
   /*Sync after each step, but use full thread capacity*/
   
   /*Create DoG filter for each sigma in set S*/
   
   /*Create Gaussian blur filter for each rho-sigma combination*/
   
   /*Convolve DoG and Gaussian filter*/
   
   /*Convolve resulting filter with input image*/
   
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

