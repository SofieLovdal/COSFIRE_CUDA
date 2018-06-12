/*sums square matrix by reduction*/
__global__ void sum(double sum, double * data, double sz) {

   const int colIdx = threadIdx.x;
   const int rowIdx = threadIdx.y;
   int linearIdx = rowIdx*sz + colIdx;

   extern __shared__ int sdata[];
   // each thread loads one element from global to shared mem
   sdata[linearIdx] = data[linearIdx];
   __syncthreads();
   // do reduction in shared mem
   for (unsigned int s=1; s < blockDim.x; s *= 2) {
      if (linearIdx % (2*s) == 0) {
	     sdata[linearIdx] += sdata[linearIdx + s];
	  }
	  __syncthreads();
   }
   // write result for this block to global mem
   if (linearIdx == 0) { 
	   sum = sdata[0];
   }	   
} 
