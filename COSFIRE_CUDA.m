function [resp timings] = COSFIRE_CUDA(image, filter1, filter2, preprocessthresh)

%This matlab script is the gateway to the CUDA implementation of the
%COSFIRE filters.
%Sofie Lovdal RUG 26.6.2018
%The preprocessing and configuration is done with the Matlab code, and the
%following steps of the algorithm are parallelized with CUDA. This script
%is called instead of BCOSFIRE_media15.m if you wish to use the GPU
%implementation instead of the Matlab version.

path(path,'./Gabor/');
path(path,'./COSFIRE/');
path(path,'./Preprocessing/');
path(path,'./Performance/');
path(path, './CUDA/');

%% Model configuration
% Prototype pattern
x = 101; y = 101; % center
line1(:, :) = zeros(201);
line1(:, x) = 1; %prototype line

% Symmetric filter params
symmfilter = cell(1);
symm_params = SystemConfig;
% COSFIRE params
symm_params.inputfilter.DoG.sigmalist = filter1.sigma;
symm_params.COSFIRE.rholist = 0:2:filter1.len;
symm_params.COSFIRE.sigma0 = filter1.sigma0 / 6;
symm_params.COSFIRE.alpha = filter1.alpha / 6;
% Orientations
numoriens = 12;
symm_params.invariance.rotation.psilist = 0:pi/numoriens:pi-pi/numoriens;
% Configuration
symmfilter{1} = configureCOSFIRE(line1, round([y x]), symm_params);
% Show the structure of the COSFIRE filter
% showCOSFIREstructure(symmfilter);

% Asymmetric filter params
asymmfilter = cell(1);
asymm_params = SystemConfig;
% COSFIRE params
asymm_params.inputfilter.DoG.sigmalist = filter2.sigma;
asymm_params.COSFIRE.rholist = 0:2:filter2.len;
asymm_params.COSFIRE.sigma0 = filter2.sigma0 / 6;
asymm_params.COSFIRE.alpha = filter2.alpha / 6;
% Orientations
numoriens = 24;
asymm_params.invariance.rotation.psilist = 0:2*pi/numoriens:(2*pi)-(2*pi/numoriens);
% Configuration
asymmfilter{1} = configureCOSFIRE(line1, round([y x]), asymm_params);
asymmfilter{1}.tuples(:, asymmfilter{1}.tuples(4,:) > pi) = []; % Deletion of on side of the filter

% Show the structure of the COSFIRE operator
% showCOSFIREstructure(asymmfilter);

%% Filtering
[image mask] = preprocess(image, [], preprocessthresh);
image = 1 - image;
figure; imagesc(image); colormap(gray); axis off; axis image; title('preprocessed image');

% Apply the symmetric B-COSFIRE to the input image
% This returns the final response for each rotation
% Here: Call mex gateway function instead of applyCOSFIRE().
% Flatten input image in row major order, which is what the CUDA functions
% expect


% Optimizations:
% Remove on-device malloc
% Remove unecessary duplicate computations
% Upgrade hardware
% Hashmap for removing unnecessary computations
% Reduce kernel overhead
% Optimize memory access patterns

[nrows, ncols, ~] = size(image);
%This assumes a single sigma for all tuples
sigma1 = symmfilter{1}.tuples(2, 1);
sigma2 = asymmfilter{1}.tuples(2, 1);

%only load the rho and phi values as the tuples: sigma is same for all
tuples1 = symmfilter{1}.tuples(3:end, :);
tuples2 = asymmfilter{1}.tuples(3:end, :);

[~, numtuples1, ~] = size(tuples1);
[~, numtuples2, ~] = size(tuples2);

%retrieve the number of unique rhos in the list of tuples
uniqueRhos1 = unique(tuples1(1,:));
[~, numRhos1] = size(uniqueRhos1);

uniqueRhos2 = unique(tuples2(1,:));
[~, numRhos2] = size(uniqueRhos2);

sigmaRatio = 0.5;
threshold = 0.0;

alpha1 = symm_params.COSFIRE.alpha;
alpha2 = asymm_params.COSFIRE.alpha;

sigma0_1 = symm_params.COSFIRE.sigma0;
sigma0_2 = asymm_params.COSFIRE.sigma0;

numRotations1 = 12;
numRotations2 = 24;

rotationStep1 = pi/(numRotations1);
rotationStep2 = (2*pi)/(numRotations2);

necessaryParameters1 = [sigma1, sigmaRatio, threshold, alpha1, sigma0_1, rotationStep1, numRotations1, numRhos1];
necessaryParameters2 = [sigma2, sigmaRatio, threshold, alpha2, sigma0_2, rotationStep2, numRotations2, numRhos2];

fid = fopen('HRF_preprocessed_1.bin','w');
fwrite(fid, reshape(image.',1,[]), 'double');
fclose(fid);

tic;
[rot1 timings1] = mexWrapper(reshape(image.',1,[]), nrows, ncols, tuples1, numtuples1, necessaryParameters1, uniqueRhos1);
[rot2 timings2] = mexWrapper(reshape(image.',1,[]), nrows, ncols, tuples2, numtuples2, necessaryParameters2, uniqueRhos2);
toc;

%rot1 is a row major vector, now turn it back into a matrix
rot1 = (reshape(rot1, [ncols, nrows])).';
rot2 = (reshape(rot2, [ncols, nrows])).';

resp = rot1 + rot2;
 %deleted the oriensmap calculation: Not included in my implementation

timings = [timings1 timings2]; %Return the timings as a long vector with times for each section.
resp = rescaleImage(resp .* mask, 0, 255);
