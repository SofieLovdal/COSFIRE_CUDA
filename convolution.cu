/*
* CUDA kernel for convolution in 2D, corresponding to conv2 in Matlab
* Sofie Lovdal 5.6.2018
*/

__global__ void conv2(double * output, double * const input, unsigned int const numRows, 
					  unsigned int const numCols, double * const kernel,
					  unsigned int const height_kernel, unsigned int const width_kernel)
{
   //width = numCols, height = numRows
   
   /*global thread ID in x dimension - moving horizontally in the image*/
   const int colIdx = blockIdx.x*blockDim.x + threadIdx.x;
   /*global thread ID in y dimension - moving vertically in the image*/
   const int rowIdx = blockIdx.y*blockDim.y + threadIdx.y;
   
   int i, j, kernelIdx, imageIdx;
    
   /*make sure we are within image*/
   if(colIdx>=numCols || rowIdx >= numRows) return; 
   
   /*Linear index of pixel corresponding to current thread */
   int linearIdx = rowIdx*numCols + colIdx;
   
   int kernel_radius=height_kernel/2;
   int imageRowIdx, imageColIdx;
   
   /*Apply convolution to linarIdx (pixel that each thread should treat) */
	double sum=0.0;
	for (i = -kernel_radius; i <= kernel_radius; i++) {
		for (j = -kernel_radius; j <= kernel_radius; j++) {
			kernelIdx = width_kernel*(i+kernel_radius) + (j+kernel_radius);
			imageRowIdx = rowIdx+i;
			imageColIdx = colIdx+j;
			imageIdx = imageRowIdx*numCols + imageColIdx;
			/*zero padding at borders: top, bottom, left, right*/
			if(imageRowIdx<0 || imageRowIdx >=numRows || imageColIdx <0 || imageColIdx >= numCols ) {
				sum+=0.0;
			} else {	
				sum=sum+input[imageIdx]*kernel[kernelIdx];
			}
		}	
	}
	output[linearIdx] = sum;
}
