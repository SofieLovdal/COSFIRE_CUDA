/*returns a Gaussian 2D function via generate2DGaussian.cu (device)*/


__global__ void getGaussian(double * output, double sigma) {
    
    int sz = ceil(sigma*3) * 2 + 1;
	int linearIdx = threadIdx.y*sz + threadIdx.x;
	if(linearIdx>=sz*sz) return;
	
	generate2DGaussian(output, sigma, sz);
	
}	
