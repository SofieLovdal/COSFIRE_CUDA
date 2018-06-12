/*Generates DoG filter as a discretized matrix with length sz by sz,
 *based on a given sigma.
 *take threshold, width into account? onoff 
 * TODO: Dynamic memory allocation for g1, g2. Currently hardcoded maximum size of filter */

__device__ void generate2DGaussian(double * output, double sigma, int sz) {
	
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

__global__ void getDoG(double * output, double sigma, double sigmaratio) {
	
	int sz = ceil(sigma*3) * 2 + 1;
	
	__shared__ double g1[200];
	__shared__ double g2[200];
	
	__syncthreads();   
	
	generate2DGaussian(g1, sigma, sz);
	generate2DGaussian(g2, sigma*sigmaratio, sz);
	
	__syncthreads();

	int linearIdx = threadIdx.y*sz + threadIdx.x;
	output[linearIdx] = g2[linearIdx]-g1[linearIdx];
}
