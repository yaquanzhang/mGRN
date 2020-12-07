function mse = get_mse_ar_heavy_tail(out, expectations, variance, history_step)
% history_step: number of steps to be used in training. 
% this is required to match the trianing sets. 
% validation_beg = 70000 + history_step + 1;  
% test_beg = 85000 + history_step + 1; 
validation_beg = 70000 + history_step;  
test_beg = 85000 + history_step;
prediction_shift = 1;
expectations_shifted = expectations(:, 1:end-prediction_shift);
%% y1 y2
g_m_1 = g_expectation(expectations_shifted(3,:), expectations_shifted(4,:), variance);
g_1 = g_expectation(expectations_shifted(6,:), expectations_shifted(7,:), variance);

g_m_2 = g_expectation(expectations_shifted(10,:), expectations_shifted(11,:), variance);
g_2 = g_expectation(expectations_shifted(13,:), expectations_shifted(14,:), variance);
y2_prediction = expectations_shifted(8,:) + expectations_shifted(9,:).*g_m_2 + expectations_shifted(12,:).*g_2;

%% product
part_1 = (expectations_shifted(1,:) + expectations_shifted(5,:).*g_1).*y2_prediction;
part_2 = (expectations_shifted(8,:) + expectations_shifted(12,:).*g_2).*expectations_shifted(2,:).*g_m_1;
part_3 = g_square_expectation(expectations_shifted(3,:), expectations_shifted(4,:), expectations_shifted(10,:), expectations_shifted(11,:), variance).*expectations_shifted(2,:).*expectations_shifted(9,:);
prediction = part_1 + part_2 + part_3 ;
errors = (prediction - out(1,prediction_shift+1:end).*out(2,prediction_shift+1:end)).^2;
product_multiply_fac = 100;
errors = errors*product_multiply_fac*product_multiply_fac;
product_mse_validation = mean(errors(validation_beg:test_beg-1));
% We use end-1 to accomodate an error in the 
%       data preparation part of the Python code. 
product_mse_test = mean(errors(test_beg:(end-1)));
%% return
mse = [product_mse_validation, product_mse_test];
end