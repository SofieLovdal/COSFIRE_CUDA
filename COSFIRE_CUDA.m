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


[nrows, ncols, ~] = size(image);

reshape(image.',1,[]); %seems fine
tuples = symmfilter{1}.tuples(2:end, :);

params_ht = symmfilter{1}.params.ht
params_cosfire = symmfilter{1}.params.COSFIRE

[hej, numtuples, ~] = size(tuples);
rot1 = mexWrapper(reshape(image.',1,[]), nrows, ncols, tuples, numtuples, 0.5, 0);

rot1(1:32)
%rot1 is a row major vector, now turn it back into a matrix
rot1 = (reshape(rot1, [ncols, nrows])).';
%rot1 = applyCOSFIRE(image, symmfilter);
%rot2 = applyCOSFIRE(image, asymmfilter);

if nargout == 1 
    % The code as presented in the paper
    % Here: take the pixelwise maximum of each rotation
    %rot1 = max(rot1{1},[],3);
    %rot2 = max(rot2{1},[],3);
    %resp = rot1 + rot2;
    resp=rot1;
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
