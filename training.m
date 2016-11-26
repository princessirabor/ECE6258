function [ to t1 ] = training( fv_samples)
%training Outputs the parameters t1,to that maximizes the log likelihood
%   Detailed explanation goes here
[Y Z] = size(fv_samples);
t_init = randn(Z+1,1);
%options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
options = optimset; options.GradObj ='on';
[thetha_x,fval,exitflag,output] = fminunc(@(t)dLh(fv_samples,t),t_init,options);
to = thetha_x(Z+1);
t1 = thetha_x(1:Z);
end

