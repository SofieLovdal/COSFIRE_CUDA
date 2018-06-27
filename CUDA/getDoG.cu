/*
 * Generates Difference-of-Gaussian filter as a discretized matrix with 
 * length sz by sz, based on a given sigma.
 * 
 */

#include "generate2DGaussian.cu"

__global__ void getDoG(double * output, double sigma, double sigmaratio) {
	
	int sz = ceil(sigma*3) * 2 + 1;
	int linearIdx = threadIdx.y*sz + threadIdx.x;
	if(linearIdx>=sz*sz) return;
	
	__shared__ double g1[900];
	__shared__ double g2[900];
	
    __syncthreads();
	cudaDeviceSynchronize();  
	
	generate2DGaussian(g1, sigma, sz);
	generate2DGaussian(g2, sigma*sigmaratio, sz);
	
	__syncthreads();
	cudaDeviceSynchronize();
	
	output[linearIdx] = g2[linearIdx]-g1[linearIdx];

}
