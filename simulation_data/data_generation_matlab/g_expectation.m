function output = g_expectation(u_expectation, v_expectation, variance)
%  variance = 0.01;
% note that the input expectations are E(u), E(v),
% but the variances are Var(log(u)) and Var(log(v))
A = 4;
z_u = z_u_expectation(u_expectation, variance);
z_v = z_u_expectation(v_expectation, variance);
output = (z_u - z_v)/A;
end

function output = z_u_expectation(u_expectation, variance)
% convert u expetctaion to z_u
% u_expectation is given after taking log
u_expectation = log(u_expectation) - variance/2;
output = exp(u_expectation.^2/(2-2*variance)).*u_expectation/((1-variance)^(3/2));
end