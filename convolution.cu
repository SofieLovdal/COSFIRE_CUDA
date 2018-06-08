/*
* CUDA kernel for convolution in 2D, corresponding to conv2 in Matlab
* Sofie Lovdal 5.6.2018
*/

__global__ void conv2(double * output, double * const input, unsigned int const numRows, 
					  unsigned int const numCols, double * const kernel,
					  unsigned int const height_kernel, unsigned int const width_kernel)
{
   
   /*current pixel*/
   const int rowIdx = blockIdx.x*blockDim.x + threadIdx.x;
   const int colIdx = blockIdx.y;
   int i, j, kernelIdx, imageIdx;
   
   /*Global thread idx for 2D grid of 1D thread blocks*/	
   int blockId = blockIdx.y * gridDim.x + blockIdx.x;
   int threadId = blockId * blockDim.x + threadIdx.x;
	
   /*Compute the index of my element. Only pixels that are not border pixels
    * in comparison to the size of the filter are convoluted*/
   const unsigned int linearIdx = threadId;
   
   /*make sure we are within image*/
   if(linearIdx>=numRows*numCols) return;
   
   /*Apply convolution to linarIdx (pixel that each thread should treat)
    * Filter uses zero padding at the borders of the images*/
	double sum=0.0;
	for (i = 0; i < height_kernel; i++) {
		for (j = 0; j < width_kernel; j++) {
			kernelIdx = width_kernel*i + j;
			imageIdx = linearIdx + (j - width_kernel/2) + (i - height_kernel/2)*numCols;   //check the idx values for correctness
			/*zero padding at row/column borders*/
			if((rowIdx - i/2 < 0) || (rowIdx + i/2 >= numRows) || (colIdx - j/2 <0) || (colIdx + i/2 >=numCols)) {
				sum+=0.0;
			} else {	
				sum+=input[imageIdx]*kernel[kernelIdx];
			}
		}	
	}
	output[linearIdx] = (sum<0 ? 0: sum);
}


