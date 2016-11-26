function [ L_t1,grad] = dLh( fv_samples,t)
%dLh Summary of this function goes here
%   Detailed explanation goes here

% Argument description
%       fv_samples         feature vectors in that class
%       t                 parameter vector( coefficient of feature vector)
%       L_t1               Log Likelihood
%       grad                  gradient

% Variables
%       sigma                  variance
%       a                      control variable-eliminates exp term when set to 0
%       u                      assumed mean

% Assuming  distribution
%Reference: Pattern Recognition textbook
%pa = (1/(sqrt(2*pi))) * exp(-1*(t1 *fv_samples + t0)/2)
%p_t1 = (1/(sigma *sqrt(2*pi))) * exp(-1*(t1 - u)*a/(2 * sigma^2))

% N = no of feature vector samples in the class  equivalent to  fv_sample
% rows
% l =no of features equivalent to  fv_sample columns
N = size(fv_samples,1);
l = size(fv_samples,2);

%thetha = zeros(1,size(fv_samples,1))
sigma = 3; 
a=1;
u=1;
to= t(l+1);
t1 =t(1:l);

%Log likelihood function
%Log likelihood terms
L_pa1 = -1 * N * log(2*pi*sigma);

L_pa2 = ((t1' *fv_samples')' + to)/2;
L_pa2 = sum(L_pa2);

L_pa3 = (t1 - u)*a/(2 * sigma^2);
L_pa3 = sum(L_pa3);

L_t1 = L_pa1-L_pa2-L_pa3;


%Gradient calculation
if(nargout>1)
grad = zeros(1,l+1);

%grad = (t1 *fv_samples + t0)*fv_samples
grad1 = ((t1' *fv_samples')'  + to);
grad2 = (t1 - u)*a/(sigma^2);


%g1 =sum(grad1.*fv_samples);
g1 =sum(bsxfun(@times,grad1,fv_samples));
g2 =N *(grad2);
g_t1 = g1+ g2';

g_to = sum(grad1);

grad(1:l) =g_t1;
grad(l+1) = g_to;
end

