%Test reduction on array for generate2DGaussian kernel
sz=20;
input = ones(sz);

kernel = parallel.gpu.CUDAKernel('sum.ptx','sum.cu','sum');
kernel.ThreadBlockSize = [sz, sz, 1];

inputMatrix = gpuArray(input);
whos

sum1 = 0.0

% 3. Call feval with defined inputs.
tic;
feval(kernel,sum1, inputMatrix, sz);
sum2 = gather(sum1)
toc;