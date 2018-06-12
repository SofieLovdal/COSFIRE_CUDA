sigma=2;
sigmaRatio=0.5;
sz = ceil(sigma*3) * 2 + 1;

tic;
g1 = fspecial('gaussian',sz,sigma);    
g2 = fspecial('gaussian',sz,sigma*sigmaRatio);
G = g2 - g1
toc;

kernel = parallel.gpu.CUDAKernel('getDoG.ptx','getDoG.cu','getDoG');
kernel.ThreadBlockSize = [sz, sz, 1];

outputMatrix=zeros(size(g1));
outputMatrix=gpuArray(outputMatrix);

tic;
outputMatrix=feval(kernel,outputMatrix, sigma, sigmaRatio);
toc;

output = gather(outputMatrix);

diff = G-output

sumDiff = sum(sum(abs(diff)))