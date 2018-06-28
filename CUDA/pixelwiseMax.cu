/*
* CUDA kernel for pixelwise maximum of all orientation-specific responses
* of a COSFIRE filter.
* Sofie Lovdal 28.6.2018
*/

__global__ void pixelwiseMax(double * output, double * const input, 
					  unsigned int const numRows, unsigned int const numCols, 
					  int const numResponses)
{
   
   const int colIdx = blockIdx.x*blockDim.x + threadIdx.x;
   const int rowIdx = blockIdx.y*blockDim.y + threadIdx.y;
    
   /*make sure we are within image*/
   if(colIdx>=numCols || rowIdx >= numRows) return; 
   
   /*Pixel to consider in outputimage*/
   int linearIdx = rowIdx*numCols + colIdx;
   
   double max=0.0, value;
   int i;
   for(i=0; i<numResponses; i++) {
	   value = input[linearIdx+i*numRows*numCols];
	   max = (value > max ? value : max);
   }
   
   output[linearIdx] = max;
}
