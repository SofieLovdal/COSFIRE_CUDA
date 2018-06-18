%tests the circshift, compares it with the matlab implementation.
%Should work just fine

A = rand(12, 12)
[nrows, ncols, ~] = size(A);

rho = 2.0;
phi = pi;

deltax = round(-rho*cos(phi));
deltay = round(-rho*sin(phi));

tic;
A1 = circshift(A,deltax,2);
A1 = circshift(A1,deltay,1) %shifts the matrix two steps in y-direction
toc;

kernel = parallel.gpu.CUDAKernel('shiftPixels.ptx','shiftPixels.cu','shiftPixels');
kernel.ThreadBlockSize = [12, 12, 1];

outputMatrix=zeros(size(A));
outputMatrix=gpuArray(outputMatrix);

tic;
outputMatrix=feval(kernel,outputMatrix, A, nrows, ncols, rho, phi);
toc;

output = gather(outputMatrix)

diff = A1-output

sumDiff = sum(sum(abs(diff)))