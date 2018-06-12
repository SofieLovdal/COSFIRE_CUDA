%test sending matrix to gpu cuda kernel, perform something on it and send back
%seems to work now! beware of order of arguments feval, thread indexing in
%kernel, block dimensions so that it covers and appropriate slice etc.

image = double(imread('./data/RETINA_example/test/images/01_test.tif')) ./ 255;
image = rgb2gray(image);

inputMatrix = gpuArray(image);
output = zeros(size(inputMatrix));
outputMatrix = gpuArray(output);
size(output);

% 1. Create CUDAKernel object.
kernel = parallel.gpu.CUDAKernel('testKernel.ptx','testKernel.cu','test');

% 2. Set object properties.
[nrows, ncols, ~] = size(inputMatrix);

blockSize = 256;
kernel.ThreadBlockSize = [blockSize, 1, 3];
kernel.GridSize = [ceil(nrows/blockSize), ncols];

% 3. Call feval with defined inputs.
outputMatrix=feval(kernel,outputMatrix,inputMatrix, nrows, ncols);

output = gather(outputMatrix);

figure; imagesc(image); colormap(gray); axis off; axis image; title('original image');
figure; imagesc(output); colormap(gray); axis off; axis image; title('After GPU');

difference = double(image-output);
msgbox(['The norm of the difference is: ' num2str(norm(difference))]);

%TRY THIS WITH A GREYSCALE IMAGE AND MAKE IT DO WHAT YOU WANT IT TO DO