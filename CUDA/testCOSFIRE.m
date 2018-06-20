%Tests the COSFIRE.cu function that controls flow of kernels and puts
%everything together
sigma=2;
sigmaRatio=0.5;
sz = ceil(sigma*3) * 2 + 1;

image = double(imread('./data/RETINA_example/test/images/01_test.tif')) ./ 255;
image = rgb2gray(image);
[nrows, ncols, ~] = size(image);

tic;
g1 = fspecial('gaussian',sz,sigma);    
g2 = fspecial('gaussian',sz,sigma*sigmaRatio);
G = g2 - g1;
output = conv2(image, G);
toc;

%tuples adapted to column first order
tuples = [2.6, 2.6, 2.6, 2.6, 2.6;
          0, 2, 2, 4, 4;
          0, 1.57, 4.71, 1.57, 4.71];
      
tuple = [2, 0, 0];      

kernel = parallel.gpu.CUDAKernel('COSFIRE.ptx','COSFIRE.cu','COSFIRE_CUDA');
%one thread for each tuple - then dynamic parallelism
kernel.ThreadBlockSize = [1 1 1];

outputMatrix=zeros(size(image));
outputMatrix=gpuArray(outputMatrix);
inputMatrix=gpuArray(image);

responses=zeros(size(image));
responseMatrix=gpuArray(responses);

tic;
outputMatrix=feval(kernel,outputMatrix, inputMatrix, numrows, numcols, tuple, 1, 0.5, sigmaRatio);
toc;
%Dynamic parallelism is not supported in MATLAB CUDAKernel objects. Need to use a MEX function instead.
output2 = gather(outputMatrix);

diff = output-output2

sumDiff = sum(sum(abs(diff)))