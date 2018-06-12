/*Generates Gaussian 2D filter given a sigma and size for the discretized matrix.
 *Corresponds to fspecial('gaussian') in Matlab.
 * Should be improved by using reduction when retrieving the sum of the matrix
 * instead of all threads looping over the full matrix.*/

__global__ void generate2DGaussian(double * output, double sigma, int sz) {
	
   /*x and y coordinates of thread in kernel. The gaussian filters are 
    *small enough for the kernel to fit into a single thread block of sz*sz*/
   const int colIdx = threadIdx.x;
   const int rowIdx = threadIdx.y;
   int linearIdx = rowIdx*sz + colIdx;
   
   /*calculate distance from centre of filter*/
   int distx = abs(colIdx - sz/2);
   int disty = abs(rowIdx - sz/2);
   
   output[linearIdx] = exp(-(pow((double)(distx), 2.0)+pow((double)(disty), 2.0))/(2*(pow(sigma, 2.0))));
   
   __syncthreads();
   
   int i, j;
   double sum=0.0;
   
   /*TODO: Reduction(:+)*/
   for(i=0; i<sz; i++) {
	   for(j=0; j<sz; j++) {
		   sum += output[i*sz + j];
		}
	}
	
   output[linearIdx]/=sum;
}

