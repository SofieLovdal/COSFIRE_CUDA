/*
* CUDA kernel for convolution, corresponding to conv2 in Matlab
* Sofie Lovdal 5.6.2018
*/

__device__ int * conv2(double * const input, unsigned int const numRows, 
					  unsigned int const numCols, double * const kernel,
					  unsigned int const height_kernel, unsigned int const width_kernel)
{
   
   /*current pixel*/
   const int rowIdx = blockIdx.x*blockDim.x + threadIdx.x;
   const int colIdx = blockIdx.y;
   const int sliceIdx = threadIdx.z;
   
   /*make sure we are within image*/
   if(rowIdx>=numRows) return;
	
   /*Compute the index of my element. Only pixels that are not border pixels
    * in comparison to the size of the filter are convoluted*/
   const unsigned int linearIdx = rowIdx + colIdx*numRows + sliceIdx*numRows*numCols;
   if(linearIdx>=numRows*numCols) return;
   
   
  
   /*assign value to output*/
   output[linearIdx]=input[linearIdx];
}
