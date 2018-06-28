/*
* CUDA kernel for geometric mean for calculating response of COSFIRE filter
* Sofie Lovdal 18.6.2018
* The input is a flattened 3D array of all responses obtained from the COSFIRE
* algorithm. The argument output is a buffer for the final response, input is a 1D
* array of dimensions numResponses*rumRows*numCols.
*/

__global__ void geometricMean(double * output, double * const input, 
					  unsigned int const numRows, unsigned int const numCols, 
					  int const numResponses, double const threshold)
{
   
   const int colIdx = blockIdx.x*blockDim.x + threadIdx.x;
   const int rowIdx = blockIdx.y*blockDim.y + threadIdx.y;
    
   /*make sure we are within image*/
   if(colIdx>=numCols || rowIdx >= numRows) return; 
   
   /*Pixel to consider in outputimage*/
   int linearIdx = rowIdx*numCols + colIdx;
   
   double product=1.0;
   int i;
   for(i=0; i<numResponses; i++) {
	   product*=input[linearIdx+i*numRows*numCols];
   }
   
   double mu = pow(product, (1.0/(double)numResponses));
   output[linearIdx] = mu; //here: (no)thresholding
}
