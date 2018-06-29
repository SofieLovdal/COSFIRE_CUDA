function [resp oriensmap] = COSFIRE_CUDA(image, filter1, filter2, preprocessthresh)

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
sigma1 = symmfilter{1}.tuples(1, 1);
sigma2 = asymmfilter{1}.tules(1, 1);

%only load the rho and phi values as the tuples: sigma is same for all
tuples1 = symmfilter{1}.tuples(3:end, :);
tuples2 = asymmfilter{1}.tuples(3:end, :);

[~, numtuples1, ~] = size(tuples1);
[~, numtuples2, ~] = size(tuples2);

sigmaRatio = 0.5;
threshold = 0.0;

alpha1 = symm_params.COSFIRE.alpha;
alpha2 = asymm_params.COSFIRE.alpha;

sigma0_1 = symm_params.COSFIRE.sigma0;
sigma0_2 = asymm_params.COSFIRE.sigma0;

numRotations1 = 12;
numRotations2 = 24;

rotationStep1 = numRotations1/pi;
rotationStep2 = numRotations2/pi;

necessaryParameters1 = [sigma1, sigmaRatio, threshold, alpha1, sigma0_1, rotationStep1, numRotations1];
%necessaryParameters2 = [sigmaRatio, threshold, alpha2, sigma0_2, rotationStep2, numRotations2];

rot1 = mexWrapper(reshape(image.',1,[]), nrows, ncols, tuples1, numtuples1, necessaryParameters1);
%rot2 = mexWrapper(reshape(image.',1,[]), nrows, ncols, tuples2, numtuples2, necessaryParameters2);
%rot1(500:510)
%rot1 is a row major vector, now turn it back into a matrix
rot1 = (reshape(rot1, [ncols, nrows])).';
%rot2 = (reshape(rot2, [ncols, nrows])).';
%rot1 = applyCOSFIRE(image, symmfilter);
%rot2 = applyCOSFIRE(image, asymmfilter);

if nargout == 1 
    % The code as presented in the paper
    %rot1 = max(rot1{1},[],3);
    %rot2 = max(rot2{1},[],3);
    %resp = rot1 + rot2;
    resp = rot1;
elseif nargout == 2   
    % Modified code to also give the orientation map as output
    for i = 1:size(rot1{1},3)
        resp(:,:,i) = rot1{1}(:, :, i) + max(rot2{1}(:,:,i),rot2{1}(:,:,i+12));    
    end
    [resp,oriensmap] = max(resp, [], 3);
    oriensmap = symm_params.invariance.rotation.psilist(oriensmap);
    oriensmap = oriensmap .* mask;
end

resp = rescaleImage(resp .* mask, 0, 255);
