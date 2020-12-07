function [out, parameters, expectations, fitted_expectations]  = simulation_ar(mean_array, N, variance)
n_series = size(mean_array,1);
out = zeros(n_series, N);
n_parameters = 7;
parameters = zeros(n_series*n_parameters , N);
expectations = zeros(n_series*n_parameters , N);
fitted_expectations = zeros(n_series*n_parameters , N);
burn_in = 100;  % the initial values that will be discarded
z_m = normrnd(0,1,1,N + burn_in);
for i = 1:n_series
    [temp_out, temp_parameters, temp_expectation, temp_fitted_expectation] = simulation_ar_single_series(mean_array(i, :), z_m, N, burn_in, variance);
    out(i,:) = temp_out;
    parameters((i-1)*n_parameters +1:i*n_parameters , :) = temp_parameters;
    expectations((i-1)*n_parameters +1:i*n_parameters , :) = temp_expectation;
    fitted_expectations((i-1)*n_parameters +1:i*n_parameters , :) = temp_fitted_expectation;
end
end