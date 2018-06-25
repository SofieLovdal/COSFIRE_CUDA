%tests cwrapper.c. Currently gives correct result for first 16 indices.

A = ones(20);

sigma = 2.0;
sigmaRatio=0.5;
sz = ceil(sigma*3) * 2 + 1;

g1 = fspecial('gaussian',sz,sigma);    
g2 = fspecial('gaussian',sz,sigma*sigmaRatio);

G = g2 - g1

A = conv2(A, G, 'same')

totalSum = sum(sum(A))

