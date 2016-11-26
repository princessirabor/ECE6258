function [ Label ] = classify( fv , thetha1_M, thethao_M)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
Labels={'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
M = 26; % no of classes
sigma = 10;
a=1;
u=1;
p_matrix =zeros(M,1); % Likelihood matrix

for i = 1:26
    
    p_matrix(i) = (1/(sqrt(2*pi))) * exp(-1*((thetha1_M(:,i)' *fv)' + thethao_M(:,i))/2)...
    *(1/(sigma *sqrt(2*pi))) * exp(-1*(thetha1_M(:,i) - u)*a/(2 * sigma^2));


end

[maxp, maxpindex] = max(p_matrix);
Label = Labels(maxpindex);

