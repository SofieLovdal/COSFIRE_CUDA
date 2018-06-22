/*Generates DoG filter as a discretized matrix with length sz by sz,
 *based on a given sigma.
 *take threshold, width into account? onoff 
 * TODO: Dynamic memory allocation for g1, g2. Currently hardcoded maximum size of filter
 */

#include "generate2DGaussian.cu"

__global__ void getDoG(double * output, double sigma, double sigmaratio) {
	
	int sz = ceil(sigma*3) * 2 + 1;
	int linearIdx = threadIdx.y*sz + threadIdx.x;
	if(linearIdx>=sz*sz) return;
	
	__shared__ double g1[200];
	__shared__ double g2[200];
	
	//printf("size: %d, threadIdx.x = %d, thredIdx.y = %d, gridDim.x=%d \n", sz, threadIdx.x, threadIdx.y, gridDim.x);
	
    __syncthreads();
	cudaDeviceSynchronize();  
	
	/*These calculations seem to go fine*/
	generate2DGaussian(g1, sigma, sz);
	generate2DGaussian(g2, sigma*sigmaratio, sz);
	
	__syncthreads();
	cudaDeviceSynchronize();
	
	output[linearIdx] = g2[linearIdx]-g1[linearIdx];
	//printf("output DoG: linearIdx = %d, output[linearIdx]=%f\n", linearIdx, output[linearIdx]);	

}
