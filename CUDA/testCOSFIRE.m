%Tests the COSFIRE.cu function that controls flow of kernels and puts
%everything together
sigma=2;
sigmaRatio=0.5;
sz = ceil(sigma*3) * 2 + 1;

%image = double(imread('./data/RETINA_example/test/images/01_test.tif')) ./ 255;
%image = rgb2gray(image);
image = ones(20);
[nrows, ncols, ~] = size(image)

tic;
g1 = fspecial('gaussian',sz,sigma);    
g2 = fspecial('gaussian',sz,sigma*sigmaRatio);
G = g2 - g1;
output = conv2(image, G, 'same');
toc;

%tuples adapted to column first order
%tuples = [2.6, 2.6, 2.6, 2.6, 2.6;
         % 0, 2, 2, 4, 4;
          %0, 1.57, 4.71, 1.57, 4.71];
      
tuple = [2, 2, 2];      
threshold = 0.5;

tic;
outputMatrix=mexWrapper(image, nrows, ncols, tuple, 1, sigmaRatio, threshold);
toc;
size(output)
size(outputMatrix)
sumGPUOutput = sum(outputMatrix)
%B = reshape(outputMatrix,[nrows,ncols]);
%diff = output-B;

%sumDiff = sum(sum(abs(diff)))

elementdiff = output(1:1, 1:nrows)-outputMatrix(1:ncols);

%TODO: figure out how matrices are loaded in and out of function to make a
%comparison between ref and CUDA version. Note: everything is 0 from
%outputMatrix currently. Perhaps output not linked properly in mex
%function?