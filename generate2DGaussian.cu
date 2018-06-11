/*Generates Gaussian filter given a sigma and size for the discretized matrix.
  Corresponds to fspecial(gaussian) in Matlab*/

__device__ double * generate2DGaussian(double sigma, int sz) {
	
	double *G = malloc(sz*sz*sizeof(double));
	
	/*calculate values according to fspecial*/
	
	return G;
}
