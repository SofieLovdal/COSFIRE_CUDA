g1 = fspecial('gaussian',17,2);    
g2 = fspecial('gaussian',17,4);

G = g2 - g1;  

img = rand(500);

% 1. Create CUDAKernel object.
kernel = parallel.gpu.CUDAKernel('convolution.ptx','convolution.cu','conv2');

% 2. Set object properties.
[nrows, ncols, ~] = size(img);
[nrowsKernel, ncolsKernel, ~] = size(G);

blockSize = 16*16;
kernel.ThreadBlockSize = [16, 16, 1];
kernel.GridSize = [ceil(nrows/16), ceil(ncols/16)];

imgGPU = gpuArray(img);
GGPU = gpuArray(G);

outputMatrix=zeros(size(img));
outputMatrix=gpuArray(outputMatrix);
% 3. Call feval with defined inputs.
outputMatrix=feval(kernel,outputMatrix,imgGPU, nrows, ncols, GGPU, nrowsKernel, ncolsKernel);

output = gather(outputMatrix);

output2 = conv2(img, G, 'same');

diff = output2-output;
diff(1)

totalDiff = sum(sum(abs(diff)))