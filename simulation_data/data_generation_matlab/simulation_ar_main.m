mean_array_full = [0.0256	0.3295	1	1.7909	0.6071	2.0504	1.6992	;
-0.0223	0.3276	1.8424	1.8344	0.6043	1.7252	1.494	;
0.0653	0.3882	1	1.9623	0.5947	1.9417	1.663	;
0.0368	0.3322	1	1.9017	0.5898	1.7723	1.5853	;
0.0075	0.3557	1.6818	2.0383	0.5651	1.9214	1.6551	;
-0.0231	0.3644	1.7973	1.8581	0.5576	1.934	1.6005	;
0.0686	0.433	1	1.9335	0.5216	2.0673	1.8113	;
0.041	0.424	1	1.6414	0.5443	1.7529	1.5982	;
-0.0091	0.343	1.8771	2.0107	0.573	2.1325	1.7065	;
0.0581	0.407	1	2.0084	0.5393	1.9392	1.6606	;
0.0219	0.3828	1.4783	1.9378	0.5759	1.999	1.6679	;
0.0037	0.4023	1.8597	1.933	0.5658	1.9396	1.7124	;
-0.0083	0.3627	2.438	1.946	0.7101	3.1765	2.6941	;
0.0335	0.3829	1.375	1.9521	0.5998	2.0163	1.7129	;
-0.0237 0.3769 1.8415 1.6061 0.5488 1.8287 1.6256];
rng(1)
n_pair = 10;
n_trail_per_pair = 1;
N = 100000;
folder_name = "simulation_data";
this_folder =  fullfile(pwd, folder_name);
if ~exist(this_folder, 'dir')
       mkdir(this_folder)
else
    disp("warning: overwrite folders.")
    % pause;
end
pair_index = zeros(n_pair, 2);
% generate stock paris
for i = 1:n_pair
    this_pair = randsample(size(mean_array_full,1),2);
    if  any(all(this_pair'  == pair_index, 2))
        % the repeated pair appears at the 7th pair. previous experiements
        % are not affected. 
        this_pair = randsample(size(mean_array_full,1),2);
    end
    pair_index(i, :) = this_pair;
end
variance = 0.01; % global control of u,v variances. Required in the calculation of expectations of g. 
%%
full_constant = [];
n_joint_above_array = [];
n_joint_below_array = [];
mse_array = zeros(n_pair*n_trail_per_pair,2);
history_step = 5;
for i = 1:n_pair
    index = pair_index(i, :);
    disp(index)
    for j = 1:n_trail_per_pair
        mean_array_sub = mean_array_full(index,:);
        [out, parameters, expectations, fitted_expectations] = simulation_ar(mean_array_sub,N, variance);
        combined = [out; parameters];
        file_name = fullfile(this_folder, num2str(i) + "-" + num2str(j) + ".csv");
        writematrix(transpose(combined),file_name)
        % save the mse
        this_idnex = (i-1)*n_trail_per_pair + j;
        mse_array(this_idnex,:) = get_mse_ar_heavy_tail(out, expectations, variance, history_step);
        disp(mse_array(this_idnex,:))
    end
end
writematrix(mse_array,fullfile(pwd, "theoretical_minimum_mse.csv"))