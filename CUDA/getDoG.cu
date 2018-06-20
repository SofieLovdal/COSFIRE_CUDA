/*Generates DoG filter as a discretized matrix with length sz by sz,
 *based on a given sigma.
 *take threshold, width into account? onoff 
 * TODO: Dynamic memory allocation for g1, g2. Currently hardcoded maximum size of filter
 */

#include "generate2DGaussian.cu"

__global__ void getDoG(double * output, double sigma, double sigmaratio) {
	
	int sz = ceil(sigma*3) * 2 + 1;
	
	__shared__ double g1[400];
	__shared__ double g2[400];
	
	__syncthreads();   
	
	generate2DGaussian(g1, sigma, sz);
	generate2DGaussian(g2, sigma*sigmaratio, sz);
	
	__syncthreads();

	int linearIdx = threadIdx.y*sz + threadIdx.x;
	if(linearIdx>sz*sz) return;
	output[linearIdx] = g2[linearIdx]-g1[linearIdx];

}
