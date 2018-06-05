/*
* Test for processing images in CUDA (input and output, are processed
* as 1D arrays).
*/
__global__ void test(double * output, double * const input,
					unsigned int const numRows, unsigned int const numCols)
{
   
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
