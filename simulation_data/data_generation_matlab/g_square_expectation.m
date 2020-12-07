function output = g_square_expectation(u_1, v_1, u_2, v_2, variance)
% variance = 0.01;
% note that the input expectations are E(u), E(v),
% but the variances are Var(log(u)) and Var(log(v))
A = 4;
u_1 = log(u_1) - variance/2;
v_1 = log(v_1) - variance/2;
u_2 = log(u_2) - variance/2;
v_2 = log(v_2) - variance/2;
f_u1_u2 = z_square_u_expectation(u_1+u_2, variance*2);
f_u1_v2 = z_square_u_expectation(u_1-v_2, variance*2);
f_u2_v1 = z_square_u_expectation(u_2-v_1, variance*2);
f_v1_v2 = z_square_u_expectation(-v_1-v_2, variance*2);
f_u1 = z_square_u_expectation(u_1, variance);
f_u2 = z_square_u_expectation(u_2, variance);
f_v1 = z_square_u_expectation(-v_1, variance);
f_v2 = z_square_u_expectation(-v_2, variance);
output = (f_u1_u2 + f_u1_v2 + f_u2_v1 + f_v1_v2)/A^2 + (f_u1 + f_u2 + f_v1 + f_v2)/A + 1;
end


function output = z_square_u_expectation(u_expectation, variance)
% convert u expetctaion to z_u
% u_expectation is given after taking log
% u_expectation = log(u_expectation) - variance/2;
output = exp(u_expectation.^2/(2-2*variance)).*(1+u_expectation.^2-variance)/((1-variance)^(5/2));
end