function [out, parameters, expectations, fitted_expectations] = simulation_ar_single_series(mean_vector, z_m, N, burn_in, variance)
%%
ar_array = {0.9, -0.8, 0.7, -0.6, 0.5};
%%
raw_ar = arima(size(ar_array,2), 0, 0);
if size(ar_array,2)>0
    raw_ar.AR = ar_array;
end
raw_ar.Variance = variance; 
N_models = 7; 
model_cell = cell(N_models, 1);
raw_ar.Constant = mean_vector(1)*(1-sum([ar_array{:}]));
model_cell{1} = raw_ar; % alpha
mean_vector(2) = mean_vector(2)/10;
mean_vector(5) = mean_vector(5)/10;
%%
% the log processes of other parameters are ARMA(5)
for i = 2:N_models
    % The unconditional variance of AR model is complicated
    % for simplicity, we fix Exp(E(p)). 
    raw_ar.Constant = log(mean_vector(i))*(1-sum([ar_array{:}]));
    model_cell{i} = raw_ar;
end
parameters = zeros(N_models, N+ burn_in);
expectations = zeros(N_models, N+ burn_in); % conditional expectations
fitted_expectations = zeros(N_models, N+ burn_in); % conditional expectations
for i = 1:N_models
    % disp(model_cell{i})
    parameters(i, :) = simulate(model_cell{i}, N + burn_in);
    expectations(i,:) = ar_one_step_prediction(model_cell{i}, parameters(i, :)');
%%    
    if i>=2
        % for positive parameters
        parameters(i, :) = exp(parameters(i, :));
        % the conditional distribution is normal
        expectations(i,:) = exp(expectations(i,:) + model_cell{i}.Variance/2);
    end
end
g_m = g(z_m, parameters(3, :), parameters(4, :));
z_individual = normrnd(0,1,1,N+ burn_in);
g_individual = g(z_individual, parameters(6, :), parameters(7, :));
out = parameters(1, :) + parameters(2, :).*g_m + parameters(5, :).*g_individual;
out = out(:, burn_in+1:end);
parameters = parameters(:, burn_in+1:end);
expectations = expectations(:, burn_in+1:end);
fitted_expectations = fitted_expectations(:, burn_in+1:end);
end

function results = g(z, u, v)
A = 4;
results = ((u.^(z) + v.^(-z))/A + 1).*z;
end