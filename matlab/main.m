% ECE 6258: Digital image processing 
% Benjamin Sullins
% GTID: 903232988
% Distance Learning Student
% School of Electrical and Computer Engineering 
% Georgia Instiute of Technology 
%
% Final Project
% Sign Language Translation To Text
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Date Modified : 11/22/2016 - Ben Sullins
% - Initial design implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% References
% http://cs229.stanford.edu/proj2011/KurdyumovHoNg-SignLanguageClassificationUsingWebcamImages.pdf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
fclose('all');

%% Import Training Images
disp('Importing Training Images');
tic

filePath = '../images/';
files = dir(filePath);
[numFiles null] = size(files);

% The images are not all the same size. We wil resize the images and then
% modify the algorithms to handle the discrepencies.
imagePaddSize = [200 200];

% Read out all the files. The first few inputs are not files, so we offset
% to number 3 instead.
offset = 3;
for i = offset:numFiles

    % Read image
    image = imread(strcat(filePath, files(i).name), 'jpg');
    
    % Convert to grayscale
    image = rgb2gray(image);
    image = imresize(image, imagePaddSize);
    
    % Add to image stack
    imageStack(:,:,i - (offset - 1)) = image;
    
end

[imSizeY imSizeX numFiles] = size(imageStack);

clear filePath files null null2 imagePaddSize image i offset
toc

% We want to flip the scale of our images because they are black hot.
disp('Converting Images to White Hot');
tic
for i = 1:numFiles
   imageStack(:,:,i) = intmax(class(imageStack)) - imageStack(:,:,i);
end

clear i
toc

%% Moment Calculation
% The first set of key descriptors will be the moments (inertia)
% calculations of the image
%
% References:
% http://stackoverflow.com/questions/27684478/how-do-i-calculate-the-central-moment-in-an-image
% https://en.wikipedia.org/wiki/Image_moment
% https://en.wikipedia.org/wiki/Eccentricity_(mathematics)
% http://breckon.eu/toby/teaching/dip/opencv/SimpleImageAnalysisbyMoments.pdf
    
disp('Computing Moment Calculations');
tic

% Compute up to the third moments
numMoments = 3;
imageStackMoments = zeros( (numMoments+1)^2, numFiles );

% Convert the images to doubles for computation
for i = 1:numFiles
   imageStackDoubles(:,:,i) = im2double( imageStack(:,:,i) );
end

for i = 1:numFiles
    
    sz = size( imageStackDoubles(:,:,i) );
    x = ( 1:sz(2) );
    y = ( 1:sz(1) ).';
    x = x - mean(x);
    y = y - mean(y);

    % Compute the p-q moments
    for p = 0:numMoments
        for q = 0:numMoments 
            
            % Skipping Conditions
            if (p == 1 || p == 2) && q == 3
                % Skip (1,3), (2,3)
                continue;
            elseif p == 2 && q == 2
                % Skip the (2,2) moment
                continue;
            elseif p == 3
                % Break all after (3,0)
                if q ~= 0
                    break;
                end
            end
            
            % Compute the moments
            Mpq = sum( reshape( bsxfun( @times, bsxfun( @times, imageStackDoubles(:,:,i), x.^p ), y.^q ), [], 1 ) );
            imageStackMoments( p*(numMoments+1) + q + 1,i) = Mpq;
        end
    end
    
end

% Compute the Moments
% The moments can be used to find defining descriptors of the image.
% Use the following source for a further description :
% http://breckon.eu/toby/teaching/dip/opencv/SimpleImageAnalysisbyMoments.pdf

imageOrientation    = zeros(1, numFiles);
imageEccentricity   = zeros(1, numFiles);

for i = 1:numFiles
    % Convert the moments to the Central Moments. This removes the
    % translation dependency.
    % See page 3
    m10Prime = 0;
    m01Prime = 0;
	m20Prime = imageStackMoments(2*(numMoments+1)+0+1, i) / imageStackMoments(0*(numMoments+1)+0+1, i) ...
               - (imageStackMoments(1*(numMoments+1)+0+1, i) / imageStackMoments(0*(numMoments+1)+0+1, i))^2; 
    m02Prime = imageStackMoments(0*(numMoments+1)+2+1, i) / imageStackMoments(0*(numMoments+1)+0+1, i) ...
               - (imageStackMoments(0*(numMoments+1)+1+1, i) / imageStackMoments(0*(numMoments+1)+0+1, i))^2;  
    m11Prime = imageStackMoments(1*(numMoments+1)+1+1, i) / imageStackMoments(0*(numMoments+1)+0+1, i) ...
               - (imageStackMoments(1*(numMoments+1)+0+1, i) / imageStackMoments(0*(numMoments+1)+0+1, i)) ...
               * (imageStackMoments(0*(numMoments+1)+1+1, i) / imageStackMoments(0*(numMoments+1)+0+1, i));
    
    % Compute the Orientation (Direction of the image)
    % The orientation of the object is defined as the tilt angle between the x-axes and
    % the axis, around which the object can be rotated with minimal inertia
    % Check zero condition
    if m20Prime ~= m02Prime
        imageOrientation(i) = (0.5) * atan( (2 * m11Prime) / (m20Prime - m02Prime) );
    end
    
    % Compute the Eccentricity (How elongated the image is)
    % The eccentricity ? can have values from 0 to 1. It is 0 with a perfectly round
    % object and 1 by a line shaped object.
    
    % The eccentricity ? is a better measure than the roundness ? of the object,
    % because it has a clearly defined range of values and therefore it can be compared
    % much better.
    imageEccentricity(i) = ( ((m20Prime - m02Prime)^2) - 4*(m11Prime^2) ) / (m20Prime + m02Prime)^2;
    
end

clear sz x y p q Mpq i numMoments m20Prime m02Prime m11Prime lambda1 lambda2 ...
        m10Prime m01Prime
toc

