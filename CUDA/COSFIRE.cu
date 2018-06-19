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
* 
* For now, the implementation allocates the necessary big chunks of 
* memory on host side and passes it to the algorithm governing kernel.
*/


__global__ void COSFIRE_CUDA(double * output, double * const input,
					unsigned int const numRows, unsigned int const numCols,
					double const * tuples, unsigned int const numTuples,
					double * responses, 
					double const threshold, double const sigmaratio)
{	   
   /*Maximize GPU load. Sync before output merging*/
   
   /*The dynamic parallelism of the kernel is structured as follwing: 
    * One thread for each tuple is launched from host side.
    * Thread i launches workflow for single tuple (outputresponse)
    */
    
    double * myTuple = *(tuples[threadIdx.x]);
    double * myResponse = *(responses[threadIdx.x]);
    
	//launch Kernel that inserts the DoG convoluted with input into myResponse (write this control flow kernel) + sync
	//launch Kernel that inserts the Gaussian maxblurring into another buffer (myResponse_maxBlur)? + sync
	//launch Kernel that shifts pixels from maxBlur buffer into new buffer (we can reuse myResponse now I guess)
	//master thread can launch kernel for geometricMean of myResponse, put into output.
	   
   
   
   
   /*Launch getDoG kernel for each sigma in set S!
    * The ideal amount of threads for this kernel is sz*sz, 
    *Return the 2D DoG filter which is then convolved here with input image*/
   
   /*Convolve with input -- separable filters?? */
   
   /*Create Gaussian blur filter for each rho-sigma combination*/
   
   /*Convolve (max-blurring) with each corresponding response from DoG convolution*/
   
   /*Obtain final response by inspecting subresponses (array of 2D matrices
    * and their corresponding shift info is needed)*/

}

