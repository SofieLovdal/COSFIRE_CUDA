/*Generates DoG filter as a discretized matrix with length sz by sz,
 *based on a given sigma.
 *take threshold, width into account? onoff*/

__device__ double * getDoG(double sigma, double sigmaratio) {
	
	int sz = ceil(sigma*3) * 2 + 1;
	
	/*can be launched as separate kernels*/
	double * g1 = generate2DGaussian(sigma, sz);
	double * g2 = generate2DGaussian(sigma*sigmaratio, sz);
	
	int i, j;
	/*can be done in parallel*/
	for (i=0; i<sz; i++) {
		for(j=0; j<sz; j++) {
			g2[i*sz + j] = g2[i*sz + j]-g1[i*sz + j];
		}
	}		 
	return g2;
}
