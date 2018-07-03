%tests the convolution CUDA kernel. getDoG should be fine, gives error in the
%order of 10e-16.
%convolution should also be fine, after half wave rectification gives error
%in the order of 10e-17.

image = double(imread('01_test.tif')) ./ 255;
image = rgb2gray(image);

sz = ceil(sigma*3) * 2 + 1;

g1 = fspecial('gaussian',sz,sigma);    
g2 = fspecial('gaussian',sz,sigma*sigmaRatio);
G = g2 - g1;

reference = conv2(image, G, 'same');
reference(find(reference < 0)) = 0; %half-wave rectification

inputMatrix = gpuArray(image);
output = zeros(size(inputMatrix));
outputMatrix = gpuArray(output);

% 1. Create CUDAKernel object.
kernel = parallel.gpu.CUDAKernel('convolution.ptx','convolution.cu','conv2');

% 2. Set object properties.
[nrows, ncols, ~] = size(inputMatrix);

kernel.ThreadBlockSize = [32, 32, 1];
kernel.GridSize = [ceil(nrows/32), ceil(ncols/32)];

G=gpuArray(G);

% 3. Call feval with defined inputs.
outputMatrix=feval(kernel,outputMatrix,inputMatrix, ncols, nrows, G, sz, sz);

output = gather(outputMatrix);

%figure; imagesc(image); colormap(gray); axis off; axis image; title('original image');
%figure; imagesc(output); colormap(gray); axis off; axis image; title('After GPU');

difference = double(reference-output);
difference(200, :)
Norm = norm(difference)

%TRY THIS WITH A GREYSCALE IMAGE AND MAKE IT DO WHAT YOU WANT IT TO DO