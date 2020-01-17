function [output, oriensmap] = ExampleBloodVesselSegmentation( )
% Delineation of blood vessels in retinal images based on combination of BCOSFIRE filter responses
%
% Version:      v1.3 
% Author(s):    Nicola Strisciuglio (nic.strisciuglio@gmail.com)
%               George Azzopardi
% 
% Description:
% Example application for the B-COSFIRE filters for delineation of
% elongated structures in images. This example demonstrates how to
% configure different types of filters (line and line-ending detectors) and
% how to combine their responses. 
%
% Application() returns:
%      output       A struct that contains the BCOSFIRE filters response
%                   and the final segmented image. In details
%                       - output.respimage: response of the combination of a symmetric and an
%                         asymemtric COSFIRE filters
%                       - output.segmented: the binary output image after
%                         thresholding
%      oriensmap    [optional] map of the orientation that gives the strongest 
%                   response at each pixel. 
%
% If you use this software please cite the following paper:
%
% "George Azzopardi, Nicola Strisciuglio, Mario Vento, Nicolai Petkov, 
% Trainable COSFIRE filters for vessel delineation with application to retinal images, 
% Medical Image Analysis, Volume 19 , Issue 1 , 46 - 57, ISSN 1361-8415
%

% Requires the compiling of the mex-file for the fast implementation of the
% max-blurring function in case it is not compiled already


if ~exist('./COSFIRE/dilate')
    BeforeUsing();
end

% Example with an image from DRIVE data set
for i=1:9
   images{i} = double(imread(strcat('./HRF/healthy/0', num2str(i, '%d'), '_h.jpg'))) ./ 255;
   %ground_truth{i} = double(imread(strcat('./DRIVE/test/2nd_manual/0', num2str(i, '%d'), + '_manual2.gif'))) ./ 255;
end

for i=10:15
   images{i} = double(imread(strcat('./HRF/healthy/', num2str(i, '%d'), '_h.jpg'))) ./ 255;
   %ground_truth{i} = double(imread(strcat('./DRIVE/test/2nd_manual/', num2str(i, '%d'), '_manual2.gif'))) ./ 255;
end    

%% Symmetric filter params
symmfilter = struct();
symmfilter.sigma     = 3.2; %changed for HRF!!
symmfilter.len       = 8;
symmfilter.sigma0    = 3;
symmfilter.alpha     = 0.3;

%% Asymmetric filter params
asymmfilter = struct();
asymmfilter.sigma     = 2.7;
asymmfilter.len       = 13;
asymmfilter.sigma0    = 1;
asymmfilter.alpha     = 0.1;

%% Filters responses
% Tresholds values
% DRIVE -> preprocessthresh = 0.5, thresh = 37
% STARE -> preprocessthresh = 0.5, thresh = 40
% CHASE_DB1 -> preprocessthresh = 0.1, thresh = 38
output = struct();
timings=zeros(100, 10);
totalTimings=zeros(100, 1);

%run the program 100 times, 5 times per input image.
for i=1:100
if nargout == 1 || nargout == 0
    tic;
    [output.respimage timings] = COSFIRE_CUDA(images{mod(i, 14)+1}, symmfilter, asymmfilter, 0.1); %preprocessthreshold is 0.1 for CHASE and 0.5 for STARE and DRIVE
    %[output.respimage] = BCOSFIRE_media15(images{mod(i, 15)+1}, symmfilter, asymmfilter, 0.5);
    totalTime = toc;
elseif nargout == 2
    [output.respimage, oriensmap] = BCOSFIRE_media15(images{i}, symmfilter, asymmfilter, 0.5);
else
    error('ERROR: too many output arguments.');
end

output.segmented = (output.respimage > 37); %or 36?
%close all;
%clear global;
%pause(0.5);
%if nargout == 0
    %figure; imagesc(output.respimage); colormap(gray); axis off; axis image; title('B-COSFIRE response image');
    %figure; imagesc(output.segmented); colormap(gray); axis off; axis image; title('B-COSFIRE segmented image');
%end    
%stop timer
%timingsVector(i, :) = timings;
totalTimings(i) = totalTime;

percentageDone = i
%accuracies{i} = EvaluateINRIA(output.segmented, ground_truth{i});
%EvaluateINRIA(reference.ans.segmented, ground_truth)
end
%statistics = zeros(21, 3);
%for i=1:20
    %Acc = (accuracies{i}(1)+accuracies{i}(3))/(584*565);
    %Se = (accuracies{i}(1)/(accuracies{i}(1)+accuracies{i}(4)));
    %Sp = (accuracies{i}(3)/(accuracies{i}(3)+accuracies{i}(2)));
    %statistics(i, :) = [Acc Se Sp];
%end
%statistics(21, :) = [mean(statistics(1:20, 1)), mean(statistics(1:20, 2)), mean(statistics(1:20, 3))];
%statistics

%Do statistics on timings:
mean(totalTimings)
std(totalTimings)

%coreAlgorithmGPUonly = mean(timingsVector(:, 5)) + mean (timingsVector(:, 10))
%maybe dotplot or similar for individual timings.




