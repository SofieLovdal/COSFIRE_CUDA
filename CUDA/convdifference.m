%tests the kernel generate2DGaussian. This kernel is OK apart from that it
%should be improved with a reduction on the sum of the matrix instead of a
%sequential summing for each thread.

sigma=2;
sz = ceil(2*3) * 2 + 1;
tic;
g1 = fspecial('gaussian',sz,sigma);
toc;

kernel = parallel.gpu.CUDAKernel('generate2DGaussian.ptx','generate2DGaussian.cu','generate2DGaussian');
kernel.ThreadBlockSize = [sz, sz, 1];

outputMatrix=zeros(size(g1));
outputMatrix=gpuArray(outputMatrix);

tic;
outputMatrix=feval(kernel,outputMatrix, sigma, sz);
toc;

output = gather(outputMatrix);

diff = g1-output
