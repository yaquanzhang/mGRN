function prediction = ar_one_step_prediction(model, y)
phi = cell2mat(model.AR);
phi = flip(phi);
p = size(phi,2); % AR
fcn = @(sub_y) multiply_full(sub_y, phi);
tA = matlab.tall.movingWindow(fcn,p,y, 'EndPoints', 'discard') + model.Constant;
prediction = [zeros(p-1,1); tA];
end

function output = multiply_full(y, phi)
output = phi*y;
end